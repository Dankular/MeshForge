# Patches

Source files that must replace their counterparts in cloned repos. Applied automatically by `scripts/deploy.sh`.

| Patch file | Replaces | Fix |
|---|---|---|
| `TripoSG_image_process.py` | `/root/TripoSG/scripts/image_process.py` | Resize BEFORE squeeze — fixes `ValueError: spatial dimensions of [1024]` when RMBG-2.0 returns `[1,H,W]` |
| `MDM_rotation2xyz.py` | `/root/MDM/model/rotation2xyz.py` | SMPL load wrapped in try/except → graceful DummySMPL fallback when gated SMPL weights unavailable |
| `MDM_mdm.py` | `/root/MDM/model/mdm.py` | Propagates `_apply`/`train` to DummySMPL so `.to(device)` doesn't crash without SMPL |

## basicsr degradations.py (auto-applied by deploy.sh)

`basicsr 1.4.2` imports `rgb_to_grayscale` from `torchvision.transforms.functional_tensor` which was removed in torchvision 0.17+. `deploy.sh` patches this via `sed` — no separate patch file needed.

```
from torchvision.transforms.functional_tensor import rgb_to_grayscale
→
from torchvision.transforms.functional import rgb_to_grayscale
```

## MV-Adapter reference_loader.py (fish_speech scoping bug)

`fish_speech/inference_engine/reference_loader.py` has an `UnboundLocalError` due to a conditional `import torchaudio.io._load_audio_fileobj` inside an except block. Fix (applied manually or via deploy.sh):

```python
# Before
import torchaudio.io._load_audio_fileobj

# After
importlib.import_module('torchaudio.io._load_audio_fileobj')
```
