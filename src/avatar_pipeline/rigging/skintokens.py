"""SkinTokens rigging integration.

SkinTokens uses TokenRig to generate a skeleton and discrete skin tokens in
one autoregressive sequence. Its FSQ-CVAE then decodes those tokens into dense
per-vertex skinning weights.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

from avatar_pipeline.models.mesh import Mesh, RiggedMesh


_REPO_ROOT = Path(__file__).resolve().parents[3]
SKINTOKENS_REPO = _REPO_ROOT / "external" / "SkinTokens"
DEFAULT_QWEN_CONFIG = SKINTOKENS_REPO / "models" / "Qwen3-0.6B"


class SkinTokensRigger:
    """Run the official SkinTokens TokenRig and FSQ-CVAE models in memory."""

    def __init__(
        self,
        num_samples: int = 54_000,
        num_vertex_samples: int = 16_384,
        seed: int = 0,
    ) -> None:
        self.num_samples = num_samples
        self.num_vertex_samples = num_vertex_samples
        self.seed = seed
        self._model: torch.nn.Module | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pretrained(
        self,
        tokenrig_checkpoint: Path,
        skin_vae_checkpoint: Path,
        qwen_config_dir: Path | None = None,
    ) -> None:
        tokenrig_checkpoint = Path(tokenrig_checkpoint).resolve()
        skin_vae_checkpoint = Path(skin_vae_checkpoint).resolve()
        qwen_config_dir = Path(qwen_config_dir or DEFAULT_QWEN_CONFIG).resolve()

        for path in (tokenrig_checkpoint, skin_vae_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"Missing SkinTokens checkpoint: {path}")
        if not (qwen_config_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Missing Qwen3 configuration for SkinTokens: {qwen_config_dir}"
            )
        if not SKINTOKENS_REPO.is_dir():
            raise FileNotFoundError(
                f"Missing SkinTokens source checkout: {SKINTOKENS_REPO}"
            )

        repo = str(SKINTOKENS_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from src.model import tokenrig as tokenrig_module
        from src.model.tokenrig import TokenRig

        tokenrig_module.LLM_LOCAL_DIR = qwen_config_dir

        checkpoint = torch.load(
            tokenrig_checkpoint,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        model_config = deepcopy(checkpoint["hyper_parameters"]["model_config"])
        model_config["pretrained_vae"] = str(skin_vae_checkpoint)
        del checkpoint

        model = TokenRig.load_from_system_checkpoint(
            checkpoint_path=str(tokenrig_checkpoint),
            model_config=model_config,
        )
        model.eval()
        # MMGP respects this marker and preserves SkinTokens' bfloat16 contract.
        model._model_dtype = torch.bfloat16
        self._model = model

    @staticmethod
    def _normalization(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        bound_min = vertices.min(axis=0)
        bound_max = vertices.max(axis=0)
        center = (bound_min + bound_max) * 0.5
        scale = float(np.max(bound_max - bound_min) * 0.5)
        if scale <= 1e-8:
            raise ValueError("Cannot rig a degenerate mesh")
        normalized = (vertices - center) / scale
        return normalized.astype(np.float32), center.astype(np.float32), scale

    def _sample_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        repo = str(SKINTOKENS_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        from src.rig_package.utils import sample_vertex_groups
        import trimesh

        tri_mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            process=False,
            maintain_order=True,
        )
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            sampled_vertices, sampled_normals, _ = sample_vertex_groups(
                vertices=vertices,
                faces=faces,
                num_samples=self.num_samples,
                num_vertex_samples=min(self.num_vertex_samples, len(vertices)),
                vertex_normals=np.asarray(tri_mesh.vertex_normals, dtype=np.float32),
                face_normals=np.asarray(tri_mesh.face_normals, dtype=np.float32),
                shuffle=True,
            )
        finally:
            np.random.set_state(state)

        if sampled_normals is None:
            raise RuntimeError("SkinTokens mesh sampling did not produce normals")
        return (
            np.asarray(sampled_vertices, dtype=np.float32),
            np.asarray(sampled_normals, dtype=np.float32),
        )

    @staticmethod
    def _top_four_weights(weights: np.ndarray) -> np.ndarray:
        if weights.ndim != 2 or weights.shape[1] == 0:
            raise RuntimeError("SkinTokens returned invalid skin weights")
        count = min(4, weights.shape[1])
        indices = np.argpartition(-weights, kth=count - 1, axis=1)[:, :count]
        sparse = np.zeros_like(weights, dtype=np.float32)
        rows = np.arange(weights.shape[0])[:, None]
        sparse[rows, indices] = weights[rows, indices]
        sparse = np.clip(sparse, 0.0, None)
        sums = sparse.sum(axis=1, keepdims=True)
        if np.any(sums <= 1e-8):
            raise RuntimeError("SkinTokens produced vertices with no skin influence")
        return sparse / sums

    @torch.inference_mode()
    def rig(self, mesh: Mesh) -> RiggedMesh:
        if self._model is None:
            raise RuntimeError("SkinTokens checkpoints have not been loaded")
        if not torch.cuda.is_available():
            raise RuntimeError("SkinTokens inference requires an NVIDIA GPU")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        normalized, center, scale = self._normalization(vertices)
        sampled_vertices, sampled_normals = self._sample_mesh(normalized, faces)

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        result = self._model.generate(
            vertices=torch.from_numpy(sampled_vertices).to(self._device),
            normals=torch.from_numpy(sampled_normals).to(self._device),
            cls="articulation",
            max_length=2048,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.0,
            num_return_sequences=1,
            num_beams=1,
            do_sample=True,
        )

        decoded = result.detokenize_output
        sampled_skin = result.skin_pred
        if decoded is None or sampled_skin is None:
            raise RuntimeError("SkinTokens failed to decode a complete rig")

        sampled_skin_np = sampled_skin.detach().float().cpu().numpy()
        nearest = cKDTree(sampled_vertices).query(normalized, k=1)[1]
        skin_weights = self._top_four_weights(sampled_skin_np[nearest])

        joint_positions = (
            np.asarray(decoded.joints, dtype=np.float32) * scale + center
        )
        parents = np.asarray(decoded.parents, dtype=np.int32)
        names = decoded.joint_names
        if names is None or len(names) != len(joint_positions):
            names = [f"bone_{i}" for i in range(len(joint_positions))]

        return RiggedMesh(
            mesh=mesh,
            joint_names=list(names),
            joint_positions=joint_positions,
            skin_weights=skin_weights,
            joint_parents=parents,
        )
