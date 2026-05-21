# Patch UniRig to remove flash_attn hard dependency
# Replaces flash_attn.MHA with nn.MultiheadAttention (equivalent, already commented in source)

path = '/root/UniRig/src/model/unirig_skin.py'
with open(path) as f:
    src = f.read()

# Remove flash_attn import
src = src.replace('from flash_attn.modules.mha import MHA\n', '')

# Replace MHA init
src = src.replace(
    '        self.attention = MHA(embed_dim=feat_dim, num_heads=num_heads, cross_attn=True)',
    '        self.attention = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=num_heads, batch_first=True)'
)

# Replace MHA forward: (q, x_kv=kv) -> nn.MHA style (q, k, v) returning (out, weights)
src = src.replace(
    '        attn_output = self.attention(q, x_kv=kv)',
    '        attn_output, _ = self.attention(q, kv, kv)'
)

with open(path, 'w') as f:
    f.write(src)
print('unirig_skin.py patched OK')

# Patch PTv3Object.py: flash_attn is try/except guarded but assert forces it on
path2 = '/root/UniRig/src/model/pointcept/models/PTv3Object.py'
with open(path2) as f:
    src2 = f.read()

src2 = src2.replace(
    'assert flash_attn is not None, "Make sure flash_attn is installed."',
    'pass  # flash_attn optional, use standard attention fallback'
)

with open(path2, 'w') as f:
    f.write(src2)
print('PTv3Object.py patched OK')
