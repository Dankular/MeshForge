import json
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from avatar_pipeline.models.mesh import BakedTextures, RiggedMesh


class GLBExporter:
    def export(
        self, mesh: RiggedMesh, textures: BakedTextures, output_path: str
    ) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        vertices = mesh.mesh.vertices.astype(np.float32)
        faces = mesh.mesh.faces.astype(np.int64)
        uv = mesh.mesh.uvs
        if uv is None or len(uv) != len(vertices):
            x = vertices[:, 0]
            z = vertices[:, 2]
            y = vertices[:, 1]
            u = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            v = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
            uv = np.stack([u, 1.0 - v], axis=1).astype(np.float32)

        albedo_img = Image.fromarray(
            np.clip(textures.albedo * 255.0, 0, 255).astype(np.uint8), mode="RGB"
        )
        normal_img = Image.fromarray(
            np.clip((textures.normal * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8),
            mode="RGB",
        )
        ao_img = Image.fromarray(
            np.clip(textures.ambient_occlusion[:, :, 0] * 255.0, 0, 255).astype(
                np.uint8
            ),
            mode="L",
        )

        material = PBRMaterial(
            name="avatar_material",
            baseColorTexture=albedo_img,
            normalTexture=normal_img,
            metallicFactor=0.0,
            roughnessFactor=0.9,
        )
        visual = TextureVisuals(uv=uv, image=albedo_img, material=material)
        tri_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False, visual=visual
        )
        scene = trimesh.Scene(tri_mesh)
        glb_bytes = scene.export(file_type="glb")
        out_path.write_bytes(glb_bytes)

        ao_path = out_path.with_suffix(".ao.png")
        normal_path = out_path.with_suffix(".normal.png")
        albedo_path = out_path.with_suffix(".albedo.png")
        ao_img.save(ao_path)
        normal_img.save(normal_path)
        albedo_img.save(albedo_path)

        payload = {
            "asset": {"version": "2.0", "generator": "avatar-pipeline"},
            "mesh": {
                "vertex_count": int(mesh.mesh.vertices.shape[0]),
                "face_count": int(mesh.mesh.faces.shape[0]),
                "semantic_regions": mesh.mesh.semantic_regions,
                "bounds_min": np.min(mesh.mesh.vertices, axis=0).astype(float).tolist(),
                "bounds_max": np.max(mesh.mesh.vertices, axis=0).astype(float).tolist(),
            },
            "rig": {
                "joint_names": mesh.joint_names,
                "joint_count": len(mesh.joint_names),
                "joint_positions": mesh.joint_positions.astype(float).tolist(),
                "joint_parents": (
                    mesh.joint_parents.astype(int).tolist()
                    if mesh.joint_parents is not None
                    else None
                ),
            },
            "textures": {
                "albedo_shape": list(textures.albedo.shape),
                "normal_shape": list(textures.normal.shape),
                "ao_shape": list(textures.ambient_occlusion.shape),
                "albedo_mean": float(np.mean(textures.albedo)),
                "normal_mean_z": float(np.mean(textures.normal[:, :, 2])),
                "ao_mean": float(np.mean(textures.ambient_occlusion)),
                "albedo_path": str(albedo_path),
                "normal_path": str(normal_path),
                "ao_path": str(ao_path),
            },
        }
        out_path.with_suffix(".meta.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return out_path
