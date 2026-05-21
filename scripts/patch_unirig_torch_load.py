# Patch UniRig run.py to add box.Box to torch safe globals
# (PyTorch 2.6+ changed weights_only default to True, breaking checkpoint loads)

path = '/root/UniRig/run.py'
with open(path) as f:
    src = f.read()

patch = """import torch
try:
    from box.box import Box as _Box
    torch.serialization.add_safe_globals([_Box])
except Exception:
    pass
"""

# Insert after the first import block (after the first set of imports)
# Find 'from box import Box' line or just prepend after the first import
if 'add_safe_globals' not in src:
    # Insert right at the top before existing imports
    src = patch + src
    with open(path, 'w') as f:
        f.write(src)
    print('run.py patched: box.Box added to torch safe globals')
else:
    print('run.py already patched')

# Also patch lightning cloud_io.py as a belt-and-suspenders fix
import os
import glob
cloud_io_paths = glob.glob(
    '/root/miniconda/envs/unirig/lib/python*/site-packages/lightning/fabric/utilities/cloud_io.py'
)
for p in cloud_io_paths:
    with open(p) as f:
        s = f.read()
    if 'weights_only=True' in s:
        s = s.replace('weights_only=True', 'weights_only=False')
        with open(p, 'w') as f:
            f.write(s)
        print(f'Patched {p}: weights_only=False')
    elif 'weights_only' not in s:
        print(f'{p}: no weights_only arg found, may be fine already')
    else:
        print(f'{p}: already patched or different version')
