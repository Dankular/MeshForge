import numpy as np

from avatar_pipeline.baking.ao_bake import AmbientOcclusionBaker
from avatar_pipeline.baking.fornos_backend import BakeRequest, FornosBackend
from avatar_pipeline.baking.normal_bake import NormalBaker
from avatar_pipeline.baking.texture_project import TextureProjector
from avatar_pipeline.fusion.head_body_align import HeadBodyAligner
from avatar_pipeline.fusion.mesh_fusion import MeshFusion
from avatar_pipeline.fusion.pshuman_refine import PSHumanRefiner
from avatar_pipeline.generators.pshuman_head import HeadGenerator
from avatar_pipeline.generators.triposg_body import BodyGenerator
from avatar_pipeline.generators.triposr_like import TripoSRLikeGenerator
from avatar_pipeline.retopo.remesh import Remesher
from avatar_pipeline.rigging.unirig import UniRig
from avatar_pipeline.sapiens.depth import DepthEstimator
from avatar_pipeline.sapiens.human_parsing import HumanParser
from avatar_pipeline.sapiens.normals import SurfaceNormals
from avatar_pipeline.sapiens.pointmap import PointmapEstimator
from avatar_pipeline.sapiens.pose_estimation import PoseEstimator


def test_generation_fusion_and_remesh_contracts():
    body = BodyGenerator().generate(
        {"height": 1.8, "radial_segments": 16, "vertical_segments": 20}
    )
    head = HeadGenerator().generate({"radius": 0.2, "subdivisions": 1})
    aligned = HeadBodyAligner().align(body, head)
    fused = MeshFusion().fuse(body, aligned)
    remeshed = Remesher().remesh(fused)

    assert body.vertices.shape[1] == 3
    assert head.faces.shape[1] == 3
    assert fused.vertices.shape[0] > body.vertices.shape[0]
    assert remeshed.vertices.shape[0] > 0
    assert remeshed.faces.shape[0] > 0


def test_baking_and_rigging_contracts():
    mesh = BodyGenerator().generate(
        {"height": 1.75, "radial_segments": 12, "vertical_segments": 18}
    )
    uv_size = 64
    albedo_a = np.full((uv_size, uv_size, 3), 0.7, dtype=np.float32)
    albedo_b = np.full((uv_size, uv_size, 3), 0.5, dtype=np.float32)
    high = np.tile(np.array([0.2, -0.1, 0.95], dtype=np.float32), (uv_size, uv_size, 1))
    low = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (uv_size, uv_size, 1))

    projected = TextureProjector().project(albedo_a, albedo_b)
    baked_normal = NormalBaker().bake(high, low)
    ao_vertex = AmbientOcclusionBaker().bake(mesh)
    baked = FornosBackend().bake(
        BakeRequest(
            albedo=projected,
            high_normals=baked_normal,
            low_normals=low,
            ao_per_vertex=ao_vertex,
        )
    )
    rigged = UniRig().rig(mesh)

    assert baked.albedo.shape == (uv_size, uv_size, 3)
    assert baked.normal.shape == (uv_size, uv_size, 3)
    assert baked.ambient_occlusion.shape == (uv_size, uv_size, 1)
    assert rigged.skin_weights.shape[0] == mesh.vertices.shape[0]
    assert rigged.skin_weights.shape[1] == len(rigged.joint_names)
    np.testing.assert_allclose(np.sum(rigged.skin_weights, axis=1), 1.0, atol=1e-5)


def test_triposr_like_and_pshuman_refine_contracts():
    image = np.full((96, 64, 3), 210, dtype=np.uint8)
    image[12:86, 22:44] = 30

    rgba = np.concatenate(
        [image.astype(np.float32) / 255.0, np.ones((96, 64, 1), dtype=np.float32)],
        axis=2,
    )
    semantic = HumanParser().parse(rgba)
    pose = PoseEstimator().estimate(rgba)
    depth = DepthEstimator().estimate(rgba)
    normals = SurfaceNormals().estimate(rgba)
    pointmap = PointmapEstimator().estimate(depth, semantic)

    coarse = TripoSRLikeGenerator().generate(image, semantic, pose, depth, normals)
    refined = PSHumanRefiner().refine(coarse, image, semantic, depth, normals, pointmap)

    assert coarse.vertices.shape[0] > 200
    assert coarse.faces.shape[0] > 200
    assert refined.vertices.shape == coarse.vertices.shape
    assert refined.faces.shape == coarse.faces.shape
