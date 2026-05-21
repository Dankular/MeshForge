import sys, cv2
sys.path.insert(0, '/root/MV-Adapter')
import numpy as np, torch
from mvadapter.utils.mesh_utils import NVDiffRastContextWrapper, load_mesh, get_orthogonal_camera, render

glb = sys.argv[1]
out = sys.argv[2]
device = 'cuda'
ctx = NVDiffRastContextWrapper(device=device, context_type='cuda')
mesh = load_mesh(glb, rescale=True, device=device)

views = [('front',-90),('right',-180),('back',-270),('left',0)]
imgs = []
for name, az in views:
    cam = get_orthogonal_camera(elevation_deg=[0], distance=[1.8],
        left=-0.55, right=0.55, bottom=-0.55, top=0.55,
        azimuth_deg=[az], device=device)
    r = render(ctx, mesh, cam, height=512, width=384,
               render_attr=True, render_depth=False, render_normal=False, attr_background=0.15)
    img = (r.attr[0].cpu().numpy()*255).clip(0,255).astype('uint8')
    imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

grid = np.concatenate(imgs, axis=1)
cv2.imwrite(out, grid)
print(f'Saved {grid.shape[1]}x{grid.shape[0]} grid to {out}')
