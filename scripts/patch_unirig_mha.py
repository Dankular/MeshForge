# Patch unirig_skin.py: replace flash_attn.MHA with a weight-compatible shim
# The checkpoint uses flash_attn MHA weight names (Wq, Wkv, out_proj)
# nn.MultiheadAttention uses in_proj_weight — incompatible with saved checkpoints
# This shim matches flash_attn MHA's weight layout exactly

shim = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class _FlashMHACompat(nn.Module):
    """
    Drop-in for flash_attn.modules.mha.MHA.
    Matches flash_attn weight layout (Wq, Wkv, out_proj) so checkpoints load cleanly.
    Uses torch SDPA for computation.
    """
    def __init__(self, embed_dim, num_heads, cross_attn=False, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.cross_attn = cross_attn
        # Weight names must match flash_attn MHA exactly
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, x_kv=None):
        B, Sq, D = x.shape
        q = self.Wq(x)
        src = x_kv if (self.cross_attn and x_kv is not None) else x
        kv = self.Wkv(src)
        k, v = kv.chunk(2, dim=-1)
        Skv = src.shape[1]

        def _reshape(t, s):
            return t.view(B, s, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = _reshape(q, Sq), _reshape(k, Skv), _reshape(v, Skv)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, Sq, D)
        return self.out_proj(out)

# Inject into a fake flash_attn module so imports resolve
import sys, types
_fa = types.ModuleType("flash_attn")
_fa_mha = types.ModuleType("flash_attn.modules")
_fa_mha_mha = types.ModuleType("flash_attn.modules.mha")
_fa_mha_mha.MHA = _FlashMHACompat
sys.modules["flash_attn"] = _fa
sys.modules["flash_attn.modules"] = _fa_mha
sys.modules["flash_attn.modules.mha"] = _fa_mha_mha
'''

# Prepend shim to run.py so it injects the fake module before any imports
path = '/root/UniRig/run.py'
with open(path) as f:
    src = f.read()

# Remove previous patch if present
if 'add_safe_globals' in src:
    # Keep the safe_globals patch, add MHA shim before it
    src = shim + src
else:
    src = shim + src

with open(path, 'w') as f:
    f.write(src)

print('run.py patched: flash_attn MHA shim injected')

# Verify the weight names match by checking unirig_skin imports
import subprocess
result = subprocess.run(
    ['grep', '-n', 'flash_attn\|MHA\|Wq\|Wkv', '/root/UniRig/src/model/unirig_skin.py'],
    capture_output=True, text=True
)
print('unirig_skin.py relevant lines:')
print(result.stdout[:500])
