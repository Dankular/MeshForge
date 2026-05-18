# Model Validation with `-check`

The CLI now includes a `check` command to validate models for pshuman rendering.

## Usage

### Check a body model
```bash
python -m v2.cli check path/to/body_model.glb
```

### Check a head model
```bash
python -m v2.cli check path/to/head_model.glb --model-type head
```

### Auto-detect model type (from filename)
```bash
# Automatically detected as "body" (contains "body" in name)
python -m v2.cli check path/to/my_body.glb

# Automatically detected as "head" (contains "head" in name)
python -m v2.cli check path/to/my_head.glb
```

### Output as JSON
```bash
python -m v2.cli check path/to/model.glb --json
```

## What Gets Checked

### Body Model Checks
- **Anatomical landmarks**: Validates vertex distribution at key body locations (feet, hips, chest, neck, chin, crown)
- **Centering**: Checks that X and Z coordinates are properly centered (x_std, z_std values)
- **Vertex density**: Reports point counts at each location

Example output:
```
✓ Model validation passed (body)

Anatomy checks (Y range: 0.000 to 1.800):
  feet   (frac=0.01): x_mean=  -0.001 x_std=  0.015 z_mean=   0.002 z_std=  0.018 n=234
  hips   (frac=0.50): x_mean=   0.003 x_std=  0.042 z_mean=  -0.005 z_std=  0.051 n=456
  chest  (frac=0.65): x_mean=   0.001 x_std=  0.038 z_mean=   0.008 z_std=  0.045 n=389
  neck   (frac=0.75): x_mean=  -0.002 x_std=  0.028 z_mean=   0.001 z_std=  0.032 n=312
  chin   (frac=0.87): x_mean=   0.005 x_std=  0.021 z_mean=  -0.003 z_std=  0.019 n=267
  crown  (frac=0.99): x_mean=   0.002 x_std=  0.012 z_mean=   0.004 z_std=  0.015 n=145
```

### Head Model Checks
- **Centroid position**: Center of mass for each mesh in the scene
- **Y-range**: Vertical extent of the model

Example output:
```
✓ Model validation passed (head)
  head_model: centroid=[-0.002, 0.145, 0.003], y_range=[0.000, 0.350]
```

## Return Codes
- `0`: Model passed validation
- `1`: Model failed validation or file not found

## Integration with Existing Check Scripts

This replaces the standalone check scripts (`check_head.py`, `check_body.py`) with integrated CLI commands:
- Unified interface for both head and body models
- Programmatic access via the `check_model()` function
- JSON output for integration with other tools
