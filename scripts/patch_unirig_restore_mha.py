# Restore unirig_skin.py to use flash_attn MHA (will be served by our shim in run.py)
# Our earlier patch replaced flash_attn MHA with nn.MultiheadAttention which breaks
# checkpoint loading because weight names don't match

path = '/root/UniRig/src/model/unirig_skin.py'
with open(path) as f:
    src = f.read()

# Restore flash_attn import (remove it if replaced, add it back)
if 'from flash_attn.modules.mha import MHA' not in src:
    # Add after the last import line before class definitions
    src = src.replace(
        'import torch_scatter\n',
        'import torch_scatter\nfrom flash_attn.modules.mha import MHA\n'
    )

# Restore original MHA init (undo nn.MultiheadAttention replacement)
src = src.replace(
    '        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)',
    '        self.attention = MHA(embed_dim=feat_dim, num_heads=num_heads, cross_attn=True)'
)

# Restore original MHA forward call
src = src.replace(
    '        attn_output, _ = self.attention(q, kv, kv)',
    '        attn_output = self.attention(q, x_kv=kv)'
)

with open(path, 'w') as f:
    f.write(src)

print('unirig_skin.py restored to use flash_attn MHA (shim will serve it)')

# Verify
import subprocess
r = subprocess.run(['grep', '-n', 'MHA\|flash_attn\|Wq\|in_proj', path],
                   capture_output=True, text=True)
print(r.stdout)
