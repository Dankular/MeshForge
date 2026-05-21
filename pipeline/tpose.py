"""
tpose.py  —  T-pose a humanoid GLB using YOLO pose estimation.

Pipeline:
  1. Render the mesh from front view (azimuth=-90)
  2. Run YOLOv8-pose to get 17 COCO keypoints in render-space
  3. Unproject keypoints through the orthographic camera to 3D
  4. Build Blender armature with bones at detected 3D joint positions (current pose)
  5. Auto-weight skin the mesh to this armature
  6. Rotate arm/leg bones to T-pose, apply deformation, export

Usage:
    blender --background --python tpose.py -- <input.glb> <output.glb>
"""

import bpy, sys, math, mathutils, os, tempfile
import numpy as np

# ── Args ─────────────────────────────────────────────────────────────────────
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []
if len(argv) < 2:
    print("Usage: blender --background --python tpose.py -- input.glb output.glb")
    sys.exit(1)
input_glb  = argv[0]
output_glb = argv[1]

# ── Step 1: Render front view using nvdiffrast (outside Blender) ───────────────
# We do this via a subprocess call before Blender scene setup,
# using the triposg Python env which has MV-Adapter + nvdiffrast.
import subprocess, json

TRIPOSG_PYTHON = '/root/miniconda/envs/triposg/bin/python'
RENDER_SCRIPT = '/tmp/_tpose_render.py'
RENDER_OUT    = '/tmp/_tpose_front.png'
KP_OUT        = '/tmp/_tpose_kp.json'

render_code = r"""
import sys, json
sys.path.insert(0, '/root/MV-Adapter')
import numpy as np, cv2, torch
from mvadapter.utils.mesh_utils import (
    NVDiffRastContextWrapper, load_mesh, get_orthogonal_camera, render,
)

body_glb = sys.argv[1]
out_png   = sys.argv[2]

device = 'cuda'
ctx     = NVDiffRastContextWrapper(device=device, context_type='cuda')
mesh_mv = load_mesh(body_glb, rescale=True, device=device)
camera  = get_orthogonal_camera(
    elevation_deg=[0], distance=[1.8],
    left=-0.55, right=0.55, bottom=-0.55, top=0.55,
    azimuth_deg=[-90], device=device,
)
out = render(ctx, mesh_mv, camera, height=1024, width=768,
             render_attr=True, render_depth=False, render_normal=False,
             attr_background=0.5)
img_np = (out.attr[0].cpu().numpy() * 255).clip(0,255).astype('uint8')
cv2.imwrite(out_png, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
print(f"Rendered to {out_png}")
"""

with open(RENDER_SCRIPT, 'w') as f:
    f.write(render_code)

print("[tpose] Rendering front view ...")
r = subprocess.run([TRIPOSG_PYTHON, RENDER_SCRIPT, input_glb, RENDER_OUT],
                   capture_output=True, text=True)
print(r.stdout.strip()); print(r.stderr[-500:] if r.stderr else '')

# ── Step 2: YOLO pose estimation ──────────────────────────────────────────────
YOLO_SCRIPT = '/tmp/_tpose_yolo.py'
yolo_code = r"""
import sys, json
import cv2
from ultralytics import YOLO
import numpy as np

img_path = sys.argv[1]
kp_path  = sys.argv[2]

model = YOLO('yolov8n-pose.pt')
img   = cv2.imread(img_path)
H, W  = img.shape[:2]

results = model(img, verbose=False)
if not results or results[0].keypoints is None:
    print("ERROR: no person detected"); sys.exit(1)

# Pick detection with highest confidence
kps_all = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
confs   = kps_all[:, :, 2].mean(axis=1)
best    = kps_all[confs.argmax()]  # (17, 3): x, y, conf

# COCO 17 keypoints:
# 0=nose 1=left_eye 2=right_eye 3=left_ear 4=right_ear
# 5=left_shoulder 6=right_shoulder 7=left_elbow 8=right_elbow
# 9=left_wrist 10=right_wrist 11=left_hip 12=right_hip
# 13=left_knee 14=right_knee 15=left_ankle 16=right_ankle

names = ['nose','left_eye','right_eye','left_ear','right_ear',
         'left_shoulder','right_shoulder','left_elbow','right_elbow',
         'left_wrist','right_wrist','left_hip','right_hip',
         'left_knee','right_knee','left_ankle','right_ankle']

kp_dict = {}
for i, name in enumerate(names):
    x, y, c = best[i]
    kp_dict[name] = {'x': float(x)/W, 'y': float(y)/H, 'conf': float(c)}
    print(f"  {name}: ({x:.1f},{y:.1f}) conf={c:.2f}")

kp_dict['img_hw'] = [int(H), int(W)]
with open(kp_path, 'w') as f:
    json.dump(kp_dict, f)
print(f"Keypoints saved to {kp_path}")
"""

with open(YOLO_SCRIPT, 'w') as f:
    f.write(yolo_code)

print("[tpose] Running YOLO pose estimation ...")
r2 = subprocess.run([TRIPOSG_PYTHON, YOLO_SCRIPT, RENDER_OUT, KP_OUT],
                    capture_output=True, text=True)
print(r2.stdout.strip()); print(r2.stderr[-300:] if r2.stderr else '')

if not os.path.exists(KP_OUT):
    print("ERROR: YOLO failed — falling back to heuristic")
    kp_data = None
else:
    with open(KP_OUT) as f:
        kp_data = json.load(f)

# ── Step 3: Unproject render-space keypoints to 3D ────────────────────────────
# Orthographic camera: left=-0.55, right=0.55, bottom=-0.55, top=0.55
# Render: 768×1024.  NDC x = 2*(px/W)-1, ndc y = 1-2*(py/H)
# World X = ndc_x * 0.55, World Y (mesh up) = ndc_y * 0.55
# We need 3D positions in the ORIGINAL mesh coordinate space.
# After Blender GLB import, original mesh Y → Blender Z, original Z → Blender -Y

def kp_to_3d(name, z_default=0.0):
    """Convert YOLO keypoint (image fraction) → Blender 3D coords."""
    if kp_data is None or name not in kp_data:
        return None
    k = kp_data[name]
    if k['conf'] < 0.3:
        return None
    # Image coords (fractions) → NDC
    ndc_x =  2 * k['x'] - 1.0      # left→right  = mesh X
    ndc_y = -(2 * k['y'] - 1.0)    # top→bottom  = mesh Y (up)
    # Orthographic: frustum ±0.55
    mesh_x = ndc_x * 0.55
    mesh_y = ndc_y * 0.55           # this is mesh-space Y (vertical)
    # After GLB import: mesh Y → Blender Z, mesh Z → Blender -Y
    bl_x = mesh_x
    bl_z = mesh_y           # height
    bl_y = z_default        # depth (not observable from front view)
    return (bl_x, bl_y, bl_z)

# Key joint positions in Blender space
J = {}
for name in ['nose','left_shoulder','right_shoulder','left_elbow','right_elbow',
             'left_wrist','right_wrist','left_hip','right_hip',
             'left_knee','right_knee','left_ankle','right_ankle']:
    p = kp_to_3d(name)
    if p: J[name] = p

print(f"[tpose] Detected joints: {list(J.keys())}")

# ── Step 4: Set up Blender scene ──────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_glb)
bpy.context.view_layer.update()

mesh_obj = next((o for o in bpy.data.objects if o.type == 'MESH'), None)
if not mesh_obj:
    print("ERROR: no mesh"); sys.exit(1)

verts_w = np.array([mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices])
z_min, z_max = verts_w[:,2].min(), verts_w[:,2].max()
x_c = (verts_w[:,0].min() + verts_w[:,0].max()) / 2
y_c = (verts_w[:,1].min() + verts_w[:,1].max()) / 2
H_mesh = z_max - z_min

def zh(frac): return z_min + frac * H_mesh
def jv(name, fallback_frac=None, fallback_x=0.0):
    """Get joint position from YOLO or use fallback."""
    if name in J:
        x, y, z = J[name]
        return (x, y_c, z)  # use mesh y_c for depth
    if fallback_frac is not None:
        return (x_c + fallback_x, y_c, zh(fallback_frac))
    return None

# ── Step 5: Build armature in CURRENT pose ────────────────────────────────────
bpy.ops.object.armature_add(location=(x_c, y_c, zh(0.5)))
arm_obj = bpy.context.object
arm_obj.name = 'PoseRig'
arm = arm_obj.data

bpy.ops.object.mode_set(mode='EDIT')
eb = arm.edit_bones

def V(xyz): return mathutils.Vector(xyz)

def add_bone(name, head, tail, parent=None, connect=False):
    b = eb.new(name)
    b.head = V(head)
    b.tail = V(tail)
    if parent and parent in eb:
        b.parent = eb[parent]
        b.use_connect = connect
    return b

# Helper: midpoint
def mid(a, b): return tuple((a[i]+b[i])/2 for i in range(3))
def offset(p, dx=0, dy=0, dz=0): return (p[0]+dx, p[1]+dy, p[2]+dz)

# ── Spine / hips ─────────────────────────────────────────────────────────────
hip_L  = jv('left_hip',  0.48, -0.07)
hip_R  = jv('right_hip', 0.48,  0.07)
sh_L   = jv('left_shoulder',  0.77, -0.20)
sh_R   = jv('right_shoulder', 0.77,  0.20)
nose   = jv('nose', 0.92)

hips_c = mid(hip_L, hip_R) if (hip_L and hip_R) else (x_c, y_c, zh(0.48))
sh_c   = mid(sh_L, sh_R)   if (sh_L and sh_R)   else (x_c, y_c, zh(0.77))

add_bone('Hips',  hips_c, offset(hips_c, dz=H_mesh*0.08))
add_bone('Spine', hips_c, offset(hips_c, dz=(sh_c[2]-hips_c[2])*0.5), 'Hips')
add_bone('Chest', offset(hips_c, dz=(sh_c[2]-hips_c[2])*0.5), sh_c, 'Spine', True)
if nose:
    neck_z = sh_c[2] + (nose[2]-sh_c[2])*0.35
    head_z = sh_c[2] + (nose[2]-sh_c[2])*0.65
    add_bone('Neck', (x_c, y_c, neck_z), (x_c, y_c, head_z),  'Chest')
    add_bone('Head', (x_c, y_c, head_z), (x_c, y_c, nose[2]+H_mesh*0.05), 'Neck', True)
else:
    add_bone('Neck', sh_c, offset(sh_c, dz=H_mesh*0.06), 'Chest')
    add_bone('Head', offset(sh_c, dz=H_mesh*0.06), offset(sh_c, dz=H_mesh*0.14), 'Neck', True)

# ── Arms (placed at DETECTED current pose positions) ─────────────────────────
el_L  = jv('left_elbow',  0.60, -0.30)
el_R  = jv('right_elbow', 0.60,  0.30)
wr_L  = jv('left_wrist',  0.45, -0.25)
wr_R  = jv('right_wrist', 0.45,  0.25)

for side, sh, el, wr in (('L', sh_L, el_L, wr_L), ('R', sh_R, el_R, wr_R)):
    if not sh: continue
    el_pos = el if el else offset(sh, dz=-H_mesh*0.15)
    wr_pos = wr if wr else offset(el_pos, dz=-H_mesh*0.15)
    hand   = offset(wr_pos, dz=(wr_pos[2]-el_pos[2])*0.4)
    add_bone(f'UpperArm.{side}', sh,     el_pos, 'Chest')
    add_bone(f'ForeArm.{side}',  el_pos, wr_pos, f'UpperArm.{side}', True)
    add_bone(f'Hand.{side}',     wr_pos, hand,   f'ForeArm.{side}',  True)

# ── Legs ─────────────────────────────────────────────────────────────────────
kn_L  = jv('left_knee',   0.25, -0.07)
kn_R  = jv('right_knee',  0.25,  0.07)
an_L  = jv('left_ankle',  0.04, -0.06)
an_R  = jv('right_ankle', 0.04,  0.06)

for side, hp, kn, an in (('L', hip_L, kn_L, an_L), ('R', hip_R, kn_R, an_R)):
    if not hp: continue
    kn_pos = kn if kn else offset(hp, dz=-H_mesh*0.23)
    an_pos = an if an else offset(kn_pos, dz=-H_mesh*0.22)
    toe    = offset(an_pos, dy=-H_mesh*0.06, dz=-H_mesh*0.02)
    add_bone(f'UpperLeg.{side}', hp,     kn_pos, 'Hips')
    add_bone(f'LowerLeg.{side}', kn_pos, an_pos, f'UpperLeg.{side}', True)
    add_bone(f'Foot.{side}',     an_pos, toe,    f'LowerLeg.{side}', True)

bpy.ops.object.mode_set(mode='OBJECT')

# ── Step 6: Skin mesh to armature ────────────────────────────────────────────
bpy.context.view_layer.objects.active = arm_obj
mesh_obj.select_set(True)
arm_obj.select_set(True)
bpy.ops.object.parent_set(type='ARMATURE_AUTO')
print("[tpose] Auto-weights applied")

# ── Step 7: Pose arms to T-pose ───────────────────────────────────────────────
# Compute per-arm rotation: from (current elbow - shoulder) direction → horizontal ±X
bpy.context.view_layer.objects.active = arm_obj
bpy.ops.object.mode_set(mode='POSE')

pb = arm_obj.pose.bones

def set_tpose_arm(side, sh_pos, el_pos):
    if not sh_pos or not el_pos:
        return
    if f'UpperArm.{side}' not in pb:
        return
    # Current upper-arm direction in armature local space
    sx = -1 if side == 'L' else 1
    # T-pose direction: ±X horizontal
    tpose_dir = mathutils.Vector((sx, 0, 0))
    # Current bone direction (head→tail) in world space
    bone = arm_obj.data.bones[f'UpperArm.{side}']
    cur_dir = (bone.tail_local - bone.head_local).normalized()
    # Rotation needed in bone's local space
    rot_quat = cur_dir.rotation_difference(tpose_dir)
    pb[f'UpperArm.{side}'].rotation_mode = 'QUATERNION'
    pb[f'UpperArm.{side}'].rotation_quaternion = rot_quat

    # Straighten forearm along the same axis
    if f'ForeArm.{side}' in pb:
        pb[f'ForeArm.{side}'].rotation_mode = 'QUATERNION'
        pb[f'ForeArm.{side}'].rotation_quaternion = mathutils.Quaternion((1,0,0,0))

set_tpose_arm('L', sh_L, el_L)
set_tpose_arm('R', sh_R, el_R)

bpy.context.view_layer.update()
bpy.ops.object.mode_set(mode='OBJECT')

# ── Step 8: Apply armature modifier ──────────────────────────────────────────
bpy.context.view_layer.objects.active = mesh_obj
mesh_obj.select_set(True)
for mod in mesh_obj.modifiers:
    if mod.type == 'ARMATURE':
        bpy.ops.object.modifier_apply(modifier=mod.name)
        print(f"[tpose] Applied modifier: {mod.name}")
        break

bpy.data.objects.remove(arm_obj, do_unlink=True)

# ── Step 9: Export ────────────────────────────────────────────────────────────
bpy.ops.export_scene.gltf(
    filepath=output_glb, export_format='GLB',
    export_texcoords=True, export_normals=True,
    export_materials='EXPORT', use_selection=False)
print(f"[tpose] Done → {output_glb}")
