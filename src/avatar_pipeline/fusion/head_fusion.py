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
    head: trimesh.Trimesh = trimesh.load(head_mesh_path, force="mesh", process=False)
    head_colors = None
    if head.visual.kind == "vertex" and head.visual.vertex_colors is not None:
        head_colors = (
            np.asarray(head.visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
        )
    print(
        f"[head_fusion] PSHuman head: {len(head.vertices):,} verts, "
        f"vertex colors: {head_colors is not None}"
    )

    # The carve outputs ~700k faces; decimate to the same density class as
    # the TripoSG body (quadric edge collapse — the same method TripoSG's
    # own inference uses for its meshes). Vertex colors ride along.
    if len(head.faces) > head_target_faces:
        import pymeshlab

        mesh_kwargs = {
            "vertex_matrix": np.asarray(head.vertices, dtype=np.float64),
            "face_matrix": np.asarray(head.faces, dtype=np.int32),
        }
        if head_colors is not None:
            mesh_kwargs["v_color_matrix"] = np.hstack(
                [head_colors.astype(np.float64),
                 np.ones((len(head_colors), 1), dtype=np.float64)]
            )
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(**mesh_kwargs))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=head_target_faces)
        # Quadric collapse leaves non-manifold edges, bowtie vertices, and
        # duplicate faces behind, and the reference TexturePipeline's UVAtlas
        # unwrap hard-rejects all of them ("Non-manifold mesh"). Repair
        # preserves vertex positions (edge/dup repair drops faces, vertex
        # repair splits verts in place), so the position-exact head-color
        # matching downstream is unaffected. One pass is not always enough
        # (duplicate faces mask bowties from the vertex repair), so iterate
        # against the same defect predicates UVAtlas enforces.
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
            print(f"[head_fusion] head repair pass left {defects}; retrying")
        dec = ms.current_mesh()
        if head_colors is not None:
            head_colors = dec.vertex_color_matrix()[:, :3].astype(np.float32)
        head = trimesh.Trimesh(
            vertices=dec.vertex_matrix().astype(np.float32),
            faces=dec.face_matrix().astype(np.int64),
            process=False,
        )
        print(f"[head_fusion] decimated head: {len(head.vertices):,} verts")

    body_tm = trimesh.Trimesh(
        vertices=body.vertices.astype(np.float64),
        faces=body.faces.astype(np.int64),
        process=False,
    )

    verts = np.array(body_tm.vertices, dtype=np.float64)
    y = verts[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    y_thresh = y_max - (y_max - y_min) * head_fraction
    head_mask = y >= y_thresh
    if head_mask.sum() < 3:
        raise RuntimeError("No head vertices found on body mesh")
    print(
        f"[head_fusion] body head verts (top {head_fraction*100:.0f}% of Y): "
        f"{head_mask.sum():,} / {len(verts):,}"
    )

    # Retract body head verts inward so the PSHuman shell sits cleanly on top
    normals = np.asarray(body_tm.vertex_normals, dtype=np.float64)
    verts[head_mask] -= normals[head_mask] * retract_amount

    # Align PSHuman head to the body head bounding box. Both meshes are
    # upright (y-up), so match the uniform scale on the VERTICAL extent —
    # head height to head-region height — and center the boxes.
    tgt = verts[head_mask]
    tgt_min, tgt_max = tgt.min(axis=0), tgt.max(axis=0)
    tgt_ctr = (tgt_min + tgt_max) * 0.5
    src_min = head.vertices.min(axis=0)
    src_max = head.vertices.max(axis=0)
    src_ctr = (src_min + src_max) * 0.5
    scale = float(tgt_max[1] - tgt_min[1]) / float(src_max[1] - src_min[1] + 1e-9)
    head_verts = (np.asarray(head.vertices, dtype=np.float64) - src_ctr) * scale + tgt_ctr
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
