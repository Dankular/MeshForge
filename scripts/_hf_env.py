from pathlib import Path
import os

def require_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        path = Path(__file__).resolve().parents[1] / ".env"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and key.strip() == "HF_TOKEN":
                    token = value.strip().strip(chr(34)).strip(chr(39))
                    break
    if not token:
        raise RuntimeError("Set HF_TOKEN in the environment or the project .env file")
    return token
