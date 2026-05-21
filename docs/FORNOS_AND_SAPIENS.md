# Fornos and Sapiens Expansion

## Fornos

Fornos is used for:
- normal baking
- AO baking
- curvature baking
- displacement transfer
- semantic ID transfer

AI topology is unstable.
Canonical topology is stable.

Therefore:
generated meshes become source detail.

## Sapiens

Sapiens is the semantic backbone.

It provides:
- segmentation
- pose estimation
- normals
- depth
- semantic regions

Semantic regions must persist throughout the pipeline.

Every vertex should eventually know:
- skin
- hair
- cloth
- eyes
- mouth
- accessories

This improves:
- baking
- fusion
- rigging
- runtime materials
