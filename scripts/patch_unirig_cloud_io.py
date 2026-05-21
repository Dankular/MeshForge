# Force weights_only=False everywhere in lightning cloud_io.py
# PyTorch 2.6+ treats weights_only=None as True, breaking UniRig checkpoints

import glob, re

for pattern in [
    '/root/miniconda/envs/unirig/lib/python*/site-packages/lightning/fabric/utilities/cloud_io.py',
    '/root/miniconda/envs/unirig/lib/python*/site-packages/lightning/fabric/plugins/io/torch_io.py',
]:
    for path in glob.glob(pattern):
        with open(path) as f:
            src = f.read()

        # Replace any torch.load call that passes weights_only as a variable
        # with a hardcoded False
        new_src = re.sub(
            r'weights_only\s*=\s*weights_only',
            'weights_only=False',
            src
        )
        # Also catch weights_only=None
        new_src = re.sub(
            r'weights_only\s*=\s*None(?!\s*\))',
            'weights_only=False',
            new_src
        )

        if new_src != src:
            with open(path, 'w') as f:
                f.write(new_src)
            print(f'Patched {path}')
        else:
            print(f'No change needed: {path}')
