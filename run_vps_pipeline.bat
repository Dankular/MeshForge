@echo off
REM MeshForge VPS Pipeline Runner
REM Uploads image to VPS, runs pipeline, downloads result

set VPS_HOST=217.154.23.86
set VPS_USER=root
set VPS_PASS=9mGvaOPbRpv1
set INPUT_IMAGE=%1
set OUTPUT_DIR=%2

if "%INPUT_IMAGE%"=="" (
    echo Usage: run_vps_pipeline.bat "path\to\image.webp" [output_dir]
    exit /b 1
)

if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=output

mkdir "%OUTPUT_DIR%" 2>nul

set IMAGE_NAME=%~nx1

echo ==========================================
echo MeshForge VPS Pipeline
echo Input: %INPUT_IMAGE%
echo VPS: %VPS_USER%@%VPS_HOST%
echo ==========================================

REM Upload image
echo [1/4] Uploading image to VPS...
pscp -pw %VPS_PASS% "%INPUT_IMAGE%" %VPS_USER%@%VPS_HOST%:/tmp/input_image.webp
if errorlevel 1 (
    echo ERROR: Failed to upload image. Make sure PuTTY/pscp is installed.
    echo Download: https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html
    exit /b 1
)

REM Run pipeline on VPS
echo [2/4] Running pipeline on VPS (this takes ~5-10 minutes)...
plink -ssh -pw %VPS_PASS% %VPS_USER%@%VPS_HOST% -batch "cd /root/MeshForge && python -c \"
import sys
sys.path.insert(0, '/root')
from app import generate_shape, apply_texture, gradio_rig
import os

# Run pipeline
print('Stage 1/3: Shape generation...')
glb, status = generate_shape('/tmp/input_image.webp', True, 50, 5.0, 42, 1)
if not glb:
    print('FAILED:', status)
    sys.exit(1)

print('Stage 2/3: Texture + face enhancement...')
glb, mv_img, status = apply_texture(glb, '/tmp/input_image.webp', True, 'sdxl', 42, True, 0.5, 2.0)
if not glb:
    print('FAILED:', status)
    sys.exit(1)

print('Stage 3/3: Rigging...')
rigged, animated, fbx, rig_status, _, _, _ = gradio_rig('/tmp/input_image.webp', glb, False, 0.5, 2.0)

print('Pipeline complete!')
print('Outputs:')
print('  Shape:', glb)
print('  Rigged:', rigged if rigged else 'N/A')
print('  FBX:', fbx if fbx else 'N/A')
\""

if errorlevel 1 (
    echo ERROR: Pipeline failed on VPS
    exit /b 1
)

REM Download results
echo [3/4] Downloading results...
pscp -pw %VPS_PASS% %VPS_USER%@%VPS_HOST%:/tmp/triposg_output/*.glb "%OUTPUT_DIR%/" 2>nul
pscp -pw %VPS_PASS% %VPS_USER%@%VPS_HOST%:/tmp/rig_out/*.glb "%OUTPUT_DIR%/" 2>nul

REM List outputs on VPS
echo [4/4] Checking outputs...
plink -ssh -pw %VPS_PASS% %VPS_USER%@%VPS_HOST% -batch "ls -la /tmp/triposg_output/ /tmp/rig_out/ 2>/dev/null || echo 'Checking alternative paths...' && find /tmp -name '*.glb' -mmin -10 2>/dev/null"

echo ==========================================
echo Pipeline complete!
echo Check %OUTPUT_DIR% for downloaded files
echo ==========================================
pause
