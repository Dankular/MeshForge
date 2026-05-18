from pathlib import Path
import trimesh
import numpy as np


def check_head_model(model_path: str | Path) -> dict:
    """Validate a head model mesh."""
    model_path = Path(model_path)
    if not model_path.exists():
        return {"status": "error", "message": f"Model not found: {model_path}"}

    try:
        scene = trimesh.load(str(model_path))
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {e}"}

    results = []
    if isinstance(scene, trimesh.Scene):
        for name, g in scene.geometry.items():
            if isinstance(g, trimesh.Trimesh):
                centroid = g.vertices.mean(0)
                y_min, y_max = g.vertices[:, 1].min(), g.vertices[:, 1].max()
                results.append({
                    "mesh": name,
                    "centroid": centroid.round(3).tolist(),
                    "y_range": [float(y_min), float(y_max)]
                })
    else:
        centroid = scene.vertices.mean(0)
        y_min, y_max = scene.vertices[:, 1].min(), scene.vertices[:, 1].max()
        results.append({
            "mesh": model_path.stem,
            "centroid": centroid.round(3).tolist(),
            "y_range": [float(y_min), float(y_max)]
        })

    return {"status": "ok", "results": results}


def check_body_model(model_path: str | Path) -> dict:
    """Validate a body model mesh with anatomical checks."""
    model_path = Path(model_path)
    if not model_path.exists():
        return {"status": "error", "message": f"Model not found: {model_path}"}

    try:
        scene = trimesh.load(str(model_path))
    except Exception as e:
        return {"status": "error", "message": f"Failed to load model: {e}"}

    if isinstance(scene, trimesh.Scene):
        body = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        body = scene

    v = body.vertices
    y_min, y_max = v[:, 1].min(), v[:, 1].max()

    checks = [
        ('feet', 0.01),
        ('hips', 0.50),
        ('chest', 0.65),
        ('neck', 0.75),
        ('chin', 0.87),
        ('crown', 0.99)
    ]

    results = []
    for name, frac in checks:
        y = y_min + (y_max - y_min) * frac
        pts = v[(v[:, 1] > y - 0.03) & (v[:, 1] < y + 0.03)]
        if len(pts) < 3:
            continue
        results.append({
            "location": name,
            "fraction": frac,
            "x_mean": float(pts[:, 0].mean()),
            "x_std": float(pts[:, 0].std()),
            "z_mean": float(pts[:, 2].mean()),
            "z_std": float(pts[:, 2].std()),
            "points": int(len(pts))
        })

    return {"status": "ok", "y_range": [float(y_min), float(y_max)], "results": results}


def check_model(model_path: str | Path, model_type: str = "auto") -> dict:
    """Check a model, auto-detecting type if needed."""
    model_path = Path(model_path)

    if model_type == "auto":
        if "head" in model_path.name.lower():
            model_type = "head"
        elif "body" in model_path.name.lower():
            model_type = "body"
        else:
            return {"status": "error", "message": "Could not auto-detect model type. Use --model-type to specify (head/body)"}

    if model_type == "head":
        return check_head_model(model_path)
    elif model_type == "body":
        return check_body_model(model_path)
    else:
        return {"status": "error", "message": f"Unknown model type: {model_type}"}
