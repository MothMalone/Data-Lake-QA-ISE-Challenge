"""Central config: paths + role->model registry, all overridable via env.

Replaces the baseline notebook's five hardcoded /content blocks. One import site.
"""
import os
from pathlib import Path


def _load_env(path: Path) -> None:
    """Minimal .env loader so we don't hard-depend on python-dotenv.

    Accepts both OPENROUTER_API_KEY and the .env's OPEN_ROUTER_API_KEY spelling.
    Does not overwrite variables already set in the real environment.
    """
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    except FileNotFoundError:
        pass


PROJECT_ROOT = Path(os.getenv("DLQA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
_load_env(PROJECT_ROOT / ".env")

DATA_LAKE_ROOT = Path(os.getenv("DLQA_DATA_LAKE_ROOT", PROJECT_ROOT / "data_lake"))
WORK_DIR = Path(os.getenv("DLQA_WORK_DIR", PROJECT_ROOT / ".dlqa"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

# key fix: the .env uses OPEN_ROUTER_API_KEY (underscore); accept both.
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPEN_ROUTER_API_KEY")
    or os.getenv("VLM_API_KEY")
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# role -> model. Defaults are the smoke-tested reachable cheap tier (2026-07).
MODELS = {
    "router":   os.getenv("DLQA_ROUTER_MODEL",   "deepseek/deepseek-v4-flash"),
    "coder":    os.getenv("DLQA_CODER_MODEL",    "deepseek/deepseek-v4-flash"),
    "synth":    os.getenv("DLQA_SYNTH_MODEL",    "deepseek/deepseek-v4-flash"),
    "verify":   os.getenv("DLQA_VERIFY_MODEL",   "deepseek/deepseek-v4-flash"),
    "vlm":      os.getenv("DLQA_VLM_MODEL",      "google/gemini-2.5-flash-lite"),
    "escalate": os.getenv("DLQA_ESCALATE_MODEL", "google/gemini-2.5-flash"),
}
