"""
test_animate.py
───────────────────────────────────────────────────────────────────────────────
End-to-end test: text prompt → motion → animated GLB.

Uses real HumanML3D 6D rotation data (NOT position-derived rotations).
Correct retargeting: final_local_rot = rest_r * smpl_local_q

Run locally:
    python test_animate.py

Optional: point at MoMask server for generation instead of dataset search:
    python test_animate.py --backend http://ssh4.vast.ai:8765
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RIGGED_GLB = r"C:\Users\sysadmin\Downloads\rigged_character.glb"
OUTPUT_GLB = r"C:\Users\sysadmin\Downloads\animated_character.glb"
QUERY      = "a person walks forward"

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default=None,
                    help="MoMask server URL, e.g. http://ssh4.vast.ai:8765")
parser.add_argument("--query", default=QUERY)
parser.add_argument("--frames", type=int, default=120)
args = parser.parse_args()

print("=" * 60)
print(f"Query:   {args.query}")
print(f"Input:   {RIGGED_GLB}")
print(f"Output:  {OUTPUT_GLB}")
print(f"Backend: {args.backend or 'dataset search (fallback)'}")
print("=" * 60)

# ── 1. Generate motion ────────────────────────────────────────────────────────
print("\n[1/2] Generating motion...")
from Retarget.generate import generate_motion

motion = generate_motion(
    prompt=args.query,
    backend_url=args.backend,
    num_frames=args.frames,
)
print(f"      motion shape: {motion.shape}  (expected [T, 263])")

# ── 2. Animate ────────────────────────────────────────────────────────────────
print("\n[2/2] Baking animation onto rigged GLB...")
from Retarget.animate import animate_glb

out = animate_glb(
    motion=motion,
    rigged_glb=RIGGED_GLB,
    output_glb=OUTPUT_GLB,
    fps=20,
    num_frames=args.frames,
)

size_mb = os.path.getsize(out) / 1_048_576
print(f"\nDone: {out}  ({size_mb:.1f} MB)")
print("Preview: drag the file into https://sandbox.babylonjs.com/")
