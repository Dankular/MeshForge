"""
pytorch3d_minimal.py
====================
Drop-in replacement for the pytorch3d subset used by PSHuman's project_mesh.py
and mesh_utils.py.  Uses nvdiffrast for GPU rasterization.

Implements:
  - Meshes / TexturesVertex
  - look_at_view_transform
  - FoVOrthographicCameras / OrthographicCameras (orthographic projection only)
  - RasterizationSettings / MeshRasterizer  (via nvdiffrast)
  - render_pix2faces_py3d  (compatibility shim)
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Texture / Mesh containers
# ---------------------------------------------------------------------------

class TexturesVertex:
    def __init__(self, verts_features):
        # verts_features: list of [N, C] tensors  (one per mesh in batch)
        self._feats = verts_features

    def verts_features_packed(self):
        return self._feats[0]

    def clone(self):
        return TexturesVertex([f.clone() for f in self._feats])

    def detach(self):
        return TexturesVertex([f.detach() for f in self._feats])

    def to(self, device):
        self._feats = [f.to(device) for f in self._feats]
        return self


class Meshes:
    def __init__(self, verts, faces, textures=None):
        self._verts = verts   # list of [N,3] float tensors
        self._faces = faces   # list of [F,3] long tensors
        self.textures = textures

    # ---- accessors --------------------------------------------------------
    def verts_padded(self):  return torch.stack(self._verts)
    def faces_padded(self):  return torch.stack(self._faces)
    def verts_packed(self):  return self._verts[0]
    def faces_packed(self):  return self._faces[0]
    def verts_list(self):    return self._verts
    def faces_list(self):    return self._faces

    def verts_normals_packed(self):
        v, f = self._verts[0], self._faces[0]
        v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        fn = torch.cross(v1 - v0, v2 - v0, dim=1)
        fn = F.normalize(fn, dim=1)
        vn = torch.zeros_like(v)
        for k in range(3):
            vn.scatter_add_(0, f[:, k:k+1].expand(-1, 3), fn)
        return F.normalize(vn, dim=1)

    # ---- device / copy ----------------------------------------------------
    def to(self, device):
        self._verts = [v.to(device) for v in self._verts]
        self._faces = [f.to(device) for f in self._faces]
        if self.textures is not None:
            self.textures.to(device)
        return self

    def clone(self):
        m = Meshes([v.clone() for v in self._verts],
                   [f.clone() for f in self._faces])
        if self.textures is not None:
            m.textures = self.textures.clone()
        return m

    def detach(self):
        m = Meshes([v.detach() for v in self._verts],
                   [f.detach() for f in self._faces])
        if self.textures is not None:
            m.textures = self.textures.detach()
        return m


# ---------------------------------------------------------------------------
# Camera math  (mirrors pytorch3d look_at_view_transform + Orthographic)
# ---------------------------------------------------------------------------

def _look_at_rotation(camera_pos: torch.Tensor,
                      at: torch.Tensor,
                      up: torch.Tensor) -> torch.Tensor:
    """Return (3,3) rotation matrix: world → camera."""
    z = F.normalize(camera_pos - at, dim=-1)          # cam looks along -Z
    x = F.normalize(torch.cross(up, z, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    R = torch.stack([x, y, z], dim=-1)               # columns = cam axes
    return R                                           # shape (3,3)


def look_at_view_transform(dist=1.0, elev=0.0, azim=0.0,
                            degrees=True, device="cpu"):
    """Matches pytorch3d convention exactly."""
    if degrees:
        elev = math.radians(float(elev))
        azim = math.radians(float(azim))

    # camera position in world
    cx = dist * math.cos(elev) * math.sin(azim)
    cy = dist * math.sin(elev)
    cz = dist * math.cos(elev) * math.cos(azim)
    eye = torch.tensor([[cx, cy, cz]], dtype=torch.float32, device=device)
    at  = torch.zeros(1, 3, device=device)
    up  = torch.tensor([[0, 1, 0]], dtype=torch.float32, device=device)

    # pytorch3d stores R transposed (row = cam axis in world space)
    R = _look_at_rotation(eye[0], at[0], up[0]).T.unsqueeze(0)  # (1,3,3)

    # T = camera position expressed in camera space
    T = torch.bmm(-R, eye.unsqueeze(-1)).squeeze(-1)             # (1,3)
    return R, T


class _OrthoCamera:
    """Minimal orthographic camera, matches FoVOrthographicCameras API."""
    def __init__(self, R, T, focal_length=1.0, device="cpu"):
        self.R = R.to(device)   # (B,3,3)
        self.T = T.to(device)   # (B,3)
        self.focal = float(focal_length)
        self.device = device

    def to(self, device):
        self.R = self.R.to(device)
        self.T = self.T.to(device)
        self.device = device
        return self

    def get_znear(self):
        return torch.tensor(0.01, device=self.device)

    def is_perspective(self):
        return False

    def transform_points_ndc(self, points):
        """
        points: (B, N, 3) world coords
        returns: (B, N, 3) NDC coords  (X,Y in [-1,1], Z = depth)
        """
        # world → camera
        pts_cam = torch.bmm(points, self.R) + self.T.unsqueeze(1)   # (B,N,3)
        # orthographic NDC: scale by focal, flip Y to match image convention
        ndc_x =  pts_cam[..., 0] * self.focal
        ndc_y = -pts_cam[..., 1] * self.focal   # pytorch3d flips Y
        ndc_z =  pts_cam[..., 2]
        return torch.stack([ndc_x, ndc_y, ndc_z], dim=-1)

    def _world_to_clip(self, verts: torch.Tensor) -> torch.Tensor:
        """verts: (N,3) → clip (N,4) for nvdiffrast."""
        pts_cam = (verts @ self.R[0].T) + self.T[0]   # (N,3)
        cx =  pts_cam[:, 0] * self.focal
        cy = -pts_cam[:, 1] * self.focal               # flip Y
        cz =  pts_cam[:, 2]
        w  = torch.ones_like(cz)
        return torch.stack([cx, cy, cz, w], dim=1)     # (N,4)


# Aliases used in project_mesh.py
def FoVOrthographicCameras(device="cpu", R=None, T=None,
                            min_x=-1, max_x=1, min_y=-1, max_y=1,
                            focal_length=None, **kwargs):
    fl = focal_length if focal_length is not None else 1.0 / (max_x + 1e-9)
    return _OrthoCamera(R, T, focal_length=fl, device=device)


def FoVPerspectiveCameras(device="cpu", R=None, T=None, fov=60, degrees=True, **kwargs):
    # Fallback: treat as orthographic at fov-derived scale (good enough for PSHuman)
    fl = 1.0 / math.tan(math.radians(fov / 2)) if degrees else 1.0 / math.tan(fov / 2)
    return _OrthoCamera(R, T, focal_length=fl, device=device)


OrthographicCameras = FoVOrthographicCameras


# ---------------------------------------------------------------------------
# Rasterizer  (nvdiffrast-based)
# ---------------------------------------------------------------------------

class RasterizationSettings:
    def __init__(self, image_size=512, blur_radius=0.0, faces_per_pixel=1):
        if isinstance(image_size, (list, tuple)):
            self.H, self.W = image_size[0], image_size[1]
        else:
            self.H = self.W = int(image_size)


class _Fragments:
    def __init__(self, pix_to_face):
        self.pix_to_face = pix_to_face.unsqueeze(-1)  # (1,H,W,1)


class MeshRasterizer:
    def __init__(self, cameras=None, raster_settings=None):
        self.cameras = cameras
        self.settings = raster_settings
        self._glctx = None

    def _get_ctx(self, device):
        if self._glctx is None:
            import nvdiffrast.torch as dr
            self._glctx = dr.RasterizeCudaContext(device=device)
        return self._glctx

    def __call__(self, meshes: Meshes, cameras=None):
        cam = cameras or self.cameras
        H, W = self.settings.H, self.settings.W
        device = meshes.verts_packed().device
        import nvdiffrast.torch as dr
        glctx = self._get_ctx(str(device))

        verts = meshes.verts_packed().to(device)
        faces = meshes.faces_packed().to(torch.int32).to(device)
        clip  = cam._world_to_clip(verts).unsqueeze(0)          # (1,N,4)
        rast, _ = dr.rasterize(glctx, clip, faces, resolution=(H, W))
        pix_to_face = rast[0, :, :, -1].to(torch.int32) - 1     # -1 = background
        return _Fragments(pix_to_face.unsqueeze(0))


# ---------------------------------------------------------------------------
# render_pix2faces_py3d shim  (used in get_visible_faces)
# ---------------------------------------------------------------------------

def render_pix2faces_py3d(meshes, cameras, H=512, W=512, **kwargs):
    """Returns {'pix_to_face': (1,H,W)} integer tensor of face indices (-1=bg)."""
    settings = RasterizationSettings(image_size=(H, W))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=settings)
    frags = rasterizer(meshes)
    return {"pix_to_face": frags.pix_to_face[..., 0]}  # (1,H,W)
