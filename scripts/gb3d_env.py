"""Load environment variables from a repo-root .env (CCDC licensing, external ref paths, ...).

Uses python-dotenv when available, otherwise a tiny built-in parser — so this also works in
the CCDC csd-python-api interpreter, which usually has neither python-dotenv nor rdkit.
"""
import os


def load_env(path=None, override=False):
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(os.path.dirname(here), ".env")
    try:  # prefer python-dotenv if installed
        from dotenv import load_dotenv
        load_dotenv(path, override=override)
        return
    except Exception:
        pass
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if override or k not in os.environ:
                os.environ[k] = v


def require(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"missing required env var {name!r} — set it in .env or the shell")
    return v
