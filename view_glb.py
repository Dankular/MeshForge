"""
Simple GLB viewer — serves the file over localhost and opens it in your browser.
Uses Google's <model-viewer> web component (loaded from CDN).
Handles skinning, PBR materials, vertex colors, animations.

Usage:
    python view_glb.py [path/to/file.glb]

Defaults to pipeline_output_rigged.glb in the same folder as this script.
"""

import sys
import os
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ── resolve target file ───────────────────────────────────────────────────────
script_dir = Path(__file__).parent
default_glb = script_dir / "pipeline_output_rigged.glb"
glb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_glb

if not glb_path.exists():
    print(f"File not found: {glb_path}")
    sys.exit(1)

glb_path = glb_path.resolve()
serve_dir = glb_path.parent
glb_filename = glb_path.name

print(f"Viewing: {glb_path}")

# ── write a minimal HTML viewer ───────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GLB Viewer — {glb_filename}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #1a1a1a; display: flex; flex-direction: column;
            align-items: center; height: 100vh; font-family: sans-serif; color: #ccc; }}
    h1  {{ padding: 8px; font-size: 14px; opacity: 0.6; }}
    model-viewer {{
      width: 100vw;
      height: calc(100vh - 36px);
      background-color: #2a2a2a;
    }}
  </style>
</head>
<body>
  <h1>{glb_filename}</h1>
  <model-viewer
    src="/{glb_filename}"
    alt="GLB model"
    camera-controls
    auto-rotate
    shadow-intensity="1"
    exposure="1"
    environment-image="neutral"
    style="--progress-bar-color: #4af">
  </model-viewer>
  <script type="module"
    src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js">
  </script>
</body>
</html>
"""

html_path = serve_dir / "_viewer.html"
html_path.write_text(html, encoding="utf-8")

# ── start HTTP server in background ──────────────────────────────────────────
PORT = 8765

class QuietHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(serve_dir), **kwargs)
    def log_message(self, fmt, *args):
        pass   # suppress request noise

def run_server():
    server = HTTPServer(("localhost", PORT), QuietHandler)
    server.serve_forever()

thread = threading.Thread(target=run_server, daemon=True)
thread.start()

url = f"http://localhost:{PORT}/_viewer.html"
print(f"Opening {url}")
print("Press Ctrl+C to quit.")
webbrowser.open(url)

try:
    thread.join()
except KeyboardInterrupt:
    print("\nDone.")
    html_path.unlink(missing_ok=True)
