# Avatar Pipeline Engineering Scope

## Core Philosophy

This project is a semantic human reconstruction engine.

The old projects:
- TripoSG
- PSHuman
- Sapiens
- REMBG
- UniRig

are behavioural references only.

We are rebuilding all processors internally.

## Hard Constraints

- No subprocess usage
- No CLI wrappers
- No Blender
- No external project imports
- No shell orchestration

## Pipeline

INPUT IMAGE
    ↓
preprocessing
    ↓
semantic parsing
    ↓
body generation
    ↓
head generation
    ↓
semantic mesh fusion
    ↓
canonical transfer
    ↓
UV generation
    ↓
texture baking
    ↓
rigging
    ↓
GLB export

## Canonical Strategy

AI meshes are source detail only.

Runtime assets should:
- share topology
- share UVs
- share skeletons
- share material slots

## Major Systems

preprocess/
sapiens/
generators/
fusion/
retopo/
baking/
rigging/
export/
runtime/
