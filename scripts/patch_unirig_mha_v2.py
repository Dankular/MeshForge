# Update run.py shim: add flash_attn_varlen_qkvpacked_func
# (PTv3Object uses this for variable-length attention in skin model)

path = '/root/UniRig/run.py'
with open(path) as f:
    src = f.read()

# Replace old shim with updated one that includes varlen func
old_marker = "# Inject into a fake flash_attn module"
new_shim = '''
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


def _flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, dropout_p=0., softmax_scale=None, **kwargs):
    """
    Drop-in for flash_attn.flash_attn_varlen_qkvpacked_func using torch SDPA.
    qkv: (total_tokens, 3, num_heads, head_dim)  [float16]
    cu_seqlens: (batch+1,) cumulative sequence lengths
    Returns: (total_tokens, num_heads, head_dim)
    """
    orig_dtype = qkv.dtype
    qkv = qkv.float()
    total, _, H, D = qkv.shape
    q, k, v = qkv.unbind(1)   # each: (total, H, D)
    scale = softmax_scale if softmax_scale is not None else (D ** -0.5)

    outputs = []
    batch_size = cu_seqlens.shape[0] - 1
    for i in range(batch_size):
        s, e = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
        qi = q[s:e].unsqueeze(0).transpose(1, 2)   # (1, H, L, D)
        ki = k[s:e].unsqueeze(0).transpose(1, 2)
        vi = v[s:e].unsqueeze(0).transpose(1, 2)
        dp = dropout_p if torch.is_grad_enabled() else 0.
        out = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=dp, scale=scale)
        outputs.append(out.transpose(1, 2).squeeze(0))   # (L, H, D)

    result = torch.cat(outputs, dim=0)   # (total, H, D)
    return result.to(orig_dtype)


# Inject into a fake flash_attn module
'''

# Find and replace the old shim marker
if old_marker in src:
    # Find start of shim (everything before the marker is safe_globals + existing code)
    shim_start = src.find('\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F')
    if shim_start == -1:
        shim_start = src.find('class _FlashMHACompat')
    inject_end = src.find('sys.modules["flash_attn.modules.mha"] = _fa_mha_mha\n') + len('sys.modules["flash_attn.modules.mha"] = _fa_mha_mha\n')

    before = src[:shim_start]
    after = src[inject_end:]

    inject_block = new_shim + '''import sys, types
_fa = types.ModuleType("flash_attn")
_fa.flash_attn_varlen_qkvpacked_func = _flash_attn_varlen_qkvpacked_func
_fa_mha = types.ModuleType("flash_attn.modules")
_fa_mha_mha = types.ModuleType("flash_attn.modules.mha")
_fa_mha_mha.MHA = _FlashMHACompat
sys.modules["flash_attn"] = _fa
sys.modules["flash_attn.modules"] = _fa_mha
sys.modules["flash_attn.modules.mha"] = _fa_mha_mha
'''
    src = before + inject_block + after
else:
    # Fresh inject at top
    inject_block = new_shim + '''import sys, types
_fa = types.ModuleType("flash_attn")
_fa.flash_attn_varlen_qkvpacked_func = _flash_attn_varlen_qkvpacked_func
_fa_mha = types.ModuleType("flash_attn.modules")
_fa_mha_mha = types.ModuleType("flash_attn.modules.mha")
_fa_mha_mha.MHA = _FlashMHACompat
sys.modules["flash_attn"] = _fa
sys.modules["flash_attn.modules"] = _fa_mha
sys.modules["flash_attn.modules.mha"] = _fa_mha_mha
'''
    src = inject_block + src

with open(path, 'w') as f:
    f.write(src)

print('run.py patched: flash_attn_varlen_qkvpacked_func added to shim')
