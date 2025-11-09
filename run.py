from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap dependencies, ensure the Ollama model is available, and launch the dev server."
    )
    parser.add_argument("--skip-install", action="store_true", help="Skip reinstalling requirements into the venv.")
    parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable uvicorn auto-reload.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind address for uvicorn.")
    parser.add_argument("--port", default=os.environ.get("PORT", "8000"), help="Port for uvicorn.")
    args = parser.parse_args()

    python_exe = ensure_virtualenv()
    if not args.skip_install:
        install_dependencies(python_exe)

    base_url, model = resolve_ollama_config()
    ensure_model_available(base_url, model)
    start_uvicorn(python_exe, args.host, args.port, args.reload)


def ensure_virtualenv() -> Path:
    venv_python = venv_python_path(VENV_DIR)
    if not VENV_DIR.exists():
        print("==> Creating virtual environment")
        run_subprocess([sys.executable, "-m", "venv", str(VENV_DIR)])
    if not venv_python.exists():
        raise SystemExit(f"Virtualenv python not found at {venv_python}")
    return venv_python


def install_dependencies(python_exe: Path) -> None:
    print("==> Installing project dependencies")
    run_subprocess([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    run_subprocess([str(python_exe), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])


def resolve_ollama_config() -> Tuple[str, str]:
    env_overrides = load_env_file(PROJECT_ROOT / ".env")
    base_url = os.environ.get("OLLAMA_BASE_URL") or env_overrides.get("OLLAMA_BASE_URL") or "http://localhost:11435"
    model = os.environ.get("OLLAMA_CHAT_MODEL") or env_overrides.get("OLLAMA_CHAT_MODEL") or "qwen2.5:3b-instruct"
    return base_url, model


def load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = strip_quotes(value.strip())
    return values


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def ensure_model_available(base_url: str, model: str) -> None:
    payload, error = fetch_ollama_tags(base_url)
    if error:
        print(f"!! Could not reach Ollama at {base_url}: {error}")
        print("   Start the daemon with `ollama serve` and try again.")
        raise SystemExit(2)
    if model_present(payload, model):
        print(f"==> Ollama model '{model}' is available")
        return
    print(f"==> Pulling Ollama model '{model}'")
    try:
        run_subprocess(["ollama", "pull", model])
    except FileNotFoundError as exc:
        raise SystemExit("The `ollama` CLI was not found on PATH. Install Ollama first.") from exc
    payload, error = fetch_ollama_tags(base_url)
    if error:
        raise SystemExit(f"Failed to verify Ollama after pull: {error}")
    if not model_present(payload, model):
        raise SystemExit(f"Ollama model '{model}' is still missing. Check the daemon logs and retry.")
    print(f"==> Ollama model '{model}' is ready")


def fetch_ollama_tags(base_url: str) -> Tuple[Dict[str, object] | None, str | None]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urlopen(url, timeout=30) as response:  # nosec: trusted local daemon
            payload = json.load(response)
    except (HTTPError, URLError) as exc:
        return None, str(exc)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON from Ollama: {exc}"
    return payload, None


def model_present(payload: Dict[str, object] | None, model: str) -> bool:
    if not payload:
        return False
    models = payload.get("models")
    if isinstance(models, dict):
        models = models.values()
    if not isinstance(models, Iterable) or isinstance(models, (str, bytes)):
        return False
    names = set()
    for item in models:
        if isinstance(item, dict):
            candidate = item.get("name") or item.get("model")
            if isinstance(candidate, str) and candidate:
                names.add(candidate)
    names |= {name.split(":", 1)[0] for name in list(names) if ":" in name}
    return model in names


def start_uvicorn(python_exe: Path, host: str, port: str, reload: bool) -> None:
    cmd = [str(python_exe), "-m", "uvicorn", "pka.app.main:app", "--host", host, "--port", str(port)]
    if reload:
        cmd.append("--reload")
    print("==> Starting FastAPI with uvicorn")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    subprocess.run(cmd, env=env, check=False, cwd=PROJECT_ROOT)


def venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_subprocess(cmd: Iterable[str]) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"   $ {printable}")
    subprocess.run(list(cmd), check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
