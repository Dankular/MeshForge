"""Fuse a PSHuman head-detail mesh into the TripoSG body before UV unwrap.

Pre-rig fusion (adapted from the recovered pipeline/face_transplant.py
`transplant_face_prerig`): detect the body's head region positionally,
retract those vertices slightly inward, align the PSHuman mesh to the head
bounding box, and concatenate. The combined mesh then flows through xatlas
unwrap, the MV-Adapter bake, and SkinTokens rigging as one piece, so the
head detail shares the body's UV atlas, textures, and skeleton.
"""
from __future__ import annotations

import numpy as np
import trimesh

from avatar_pipeline.models.mesh import Mesh


def uvatlas_defects(verts: np.ndarray, faces: np.ndarray) -> dict[str, int]:
    """Defect counts that make open3d's UVAtlas unwrap reject a mesh.

    UVAtlas requires full manifoldness: edge-manifold (<=2 faces per edge),
    vertex-manifold (no bowties), and no duplicate faces. Checked with
    open3d's own predicates — the exact ones compute_uvatlas enforces.
    """
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    tris_sorted = np.sort(np.asarray(faces), axis=1)
    return {
        "nm_edges": len(
            np.asarray(m.get_non_manifold_edges(allow_boundary_edges=True))
        ),
        "nm_verts": len(np.asarray(m.get_non_manifold_vertices())),
        "dup_faces": len(tris_sorted) - len(np.unique(tris_sorted, axis=0)),
    }


def _neck_line(verts: np.ndarray, n_bins: int = 100) -> float | None:
    """y of the neck: the top of the wingspan jump in the width profile.

    A fixed anatomical head fraction fails on TripoSG bodies with off
    proportions — this class of body has a vestigial head stub (top ~3%,
    width ~0.3) with the T-pose wingspan (width ~1.7) starting right below
    it, so a 13% cut centers the transplanted head at the chest and sinks
    it into the torso. Image-space pose keypoints fail for the same reason:
    the mesh does not share the image's proportions. The mesh's own width
    profile is unambiguous in a T-pose: scanning down, the horizontal
    extent multiplies at the shoulder line. Returns the mesh y of the last
    narrow bin, or None when no clear jump exists.
    """
    y = verts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    height = y_max - y_min
    if height <= 0:
        return None
    bins = np.clip(((y_max - y) / height * n_bins).astype(int), 0, n_bins - 1)
    x_min = np.full(n_bins, np.inf)
    x_max = np.full(n_bins, -np.inf)
    np.minimum.at(x_min, bins, verts[:, 0])
    np.maximum.at(x_max, bins, verts[:, 0])
    width = np.where(np.isfinite(x_min), x_max - x_min, 0.0)

    peak = float(width.max())
    for i in range(min(30, n_bins - 1)):  # neck must be in the top 30%
        if (
            width[i] > 0
            and width[i + 1] > 1.8 * width[i]
            and width[i] < 0.5 * peak
        ):
            return y_max - (i + 1) / n_bins * height
    return None


def load_decimated_colored(
    mesh_path: str, target_faces: int
) -> tuple[trimesh.Trimesh, np.ndarray | None]:
    """Load a PSHuman colored OBJ, decimated to *target_faces* and repaired
    to UVAtlas cleanliness. Vertex colors (float32 [0,1]) ride along.

    Quadric collapse leaves non-manifold edges, bowtie vertices, and
    duplicate faces behind, all of which the reference TexturePipeline's
    UVAtlas unwrap hard-rejects. The repair preserves vertex positions
    (edge/dup repair drops faces, vertex repair splits verts in place), so
    position-exact color matching downstream is unaffected. One pass is not
    always enough (duplicate faces mask bowties from the vertex repair), so
    iterate against the same defect predicates UVAtlas enforces.
    """
    mesh: trimesh.Trimesh = trimesh.load(mesh_path, force="mesh", process=False)
    colors = None
    if mesh.visual.kind == "vertex" and mesh.visual.vertex_colors is not None:
        colors = (
            np.asarray(mesh.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        )
    if len(mesh.faces) <= target_faces:
        return mesh, colors

    import pymeshlab

    mesh_kwargs = {
        "vertex_matrix": np.asarray(mesh.vertices, dtype=np.float64),
        "face_matrix": np.asarray(mesh.faces, dtype=np.int32),
    }
    if colors is not None:
        mesh_kwargs["v_color_matrix"] = np.hstack(
            [colors.astype(np.float64),
             np.ones((len(colors), 1), dtype=np.float64)]
        )
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(**mesh_kwargs))
    ms.meshing_merge_close_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    for _ in range(3):
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_repair_non_manifold_vertices()
        ms.meshing_remove_unreferenced_vertices()
        cur = ms.current_mesh()
        defects = uvatlas_defects(cur.vertex_matrix(), cur.face_matrix())
        if not any(defects.values()):
            break
        print(f"[head_fusion] repair pass left {defects}; retrying")
    dec = ms.current_mesh()
    if colors is not None:
        colors = dec.vertex_color_matrix()[:, :3].astype(np.float32)
    out = trimesh.Trimesh(
        vertices=dec.vertex_matrix().astype(np.float32),
        faces=dec.face_matrix().astype(np.int64),
        process=False,
    )
    return out, colors


def fuse_head_prerig(
    body: Mesh,
    head_mesh_path: str,
    head_fraction: float = 0.13,
    retract_amount: float = 0.004,
    head_target_faces: int = 30_000,
) -> tuple[Mesh, np.ndarray | None, int]:
    """Concatenate the PSHuman head onto the TripoSG body (both unrigged).

    The PSHuman input was already a head-only crop, so its reconstruction IS
    the head — no further cropping (verified by rendering the carve output;
    cropping it again produces a skull-cap shell). *head_fraction* is the
    body height fraction treated as the head: ~1/7.5 of standing height.

    PSHuman's per-vertex colors (its own multiview color optimization) are
    carried through decimation so the head region of the atlas can be
    textured from PSHuman's views instead of the full-figure bake.

    Returns (fused_mesh_without_UVs, head_vertex_colors | None, head_vert_start).
    """
    # The carve outputs ~700k faces; decimate to the same density class as
    # the TripoSG body (quadric edge collapse — the same method TripoSG's
    # own inference uses for its meshes). Vertex colors ride along.
    head, head_colors = load_decimated_colored(head_mesh_path, head_target_faces)
    print(
        f"[head_fusion] PSHuman head: {len(head.vertices):,} verts, "
        f"vertex colors: {head_colors is not None}"
    )

    body_tm = trimesh.Trimesh(
        vertices=body.vertices.astype(np.float64),
        faces=body.faces.astype(np.int64),
        process=False,
    )

    verts = np.array(body_tm.vertices, dtype=np.float64)
    y = verts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    height = y_max - y_min

    neck_y = _neck_line(verts)
    if neck_y is not None:
        # Anatomical placement: the body's own head is whatever sits above
        # the neck line (possibly a vestigial stub far smaller than a real
        # head — scaling the PSHuman head into that pocket gives a pinhead
        # sunk behind the collar). Scale the head to ~1/7.5 of standing
        # height instead and seat it on the neck, regardless of how little
        # head geometry the body grew.
        print(f"[head_fusion] neck line at y={neck_y:.3f} (width-profile jump)")
        head_mask = y >= neck_y
    else:
        print(
            f"[head_fusion] no neck jump found; fixed "
            f"{head_fraction:.0%} head fraction"
        )
        head_mask = y >= y_max - height * head_fraction
    if head_mask.sum() < 3:
        raise RuntimeError("No head vertices found on body mesh")
    print(
        f"[head_fusion] body head/stub verts: "
        f"{head_mask.sum():,} / {len(verts):,}"
    )

    # Retract body head verts inward so the PSHuman shell sits cleanly on top
    normals = np.asarray(body_tm.vertex_normals, dtype=np.float64)
    verts[head_mask] -= normals[head_mask] * retract_amount

    src_min = head.vertices.min(axis=0)
    src_max = head.vertices.max(axis=0)
    src_ctr = (src_min + src_max) * 0.5
    src_h = float(src_max[1] - src_min[1] + 1e-9)
    tgt = verts[head_mask]
    tgt_min, tgt_max = tgt.min(axis=0), tgt.max(axis=0)
    tgt_ctr = (tgt_min + tgt_max) * 0.5

    if neck_y is not None:
        # Anatomical head height, seated on the neck line with a 10% sink
        # for weld overlap; centered horizontally on the stub.
        dst_h = 0.13 * height
        scale = dst_h / src_h
        head_verts = (np.asarray(head.vertices, dtype=np.float64) - src_ctr) * scale
        head_verts[:, 0] += tgt_ctr[0]
        head_verts[:, 2] += tgt_ctr[2]
        head_verts[:, 1] += (neck_y - 0.10 * dst_h) - head_verts[:, 1].min()
    else:
        # Legacy: match the head-region bounding box vertically and center.
        scale = float(tgt_max[1] - tgt_min[1]) / src_h
        head_verts = (
            np.asarray(head.vertices, dtype=np.float64) - src_ctr
        ) * scale + tgt_ctr
    print(f"[head_fusion] aligned head: scale={scale:.4f}")

    combined_verts = np.vstack([verts, head_verts]).astype(np.float32)
    combined_faces = np.vstack(
        [
            body.faces.astype(np.int64),
            np.asarray(head.faces, dtype=np.int64) + len(verts),
        ]
    ).astype(np.int32)

    fused = Mesh(
        vertices=combined_verts,
        faces=combined_faces,
        semantic_regions=body.semantic_regions,
    )
    print(
        f"[head_fusion] combined: {len(combined_verts):,} verts, "
        f"{len(combined_faces):,} faces"
    )

    # The reference TexturePipeline runs UVAtlas with preprocess=False (to
    # keep vertex positions exact for head-color matching) and UVAtlas
    # hard-rejects any non-manifoldness — catch it here, where the cause is
    # attributable, not 20 minutes later inside the bake.
    defects = uvatlas_defects(combined_verts, combined_faces)
    if any(defects.values()):
        raise RuntimeError(
            f"[head_fusion] fused mesh is not UVAtlas-clean: {defects}"
        )
    return fused, head_colors, len(verts)
