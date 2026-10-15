"""
Optional `.env` support (no dependency): `KEY=value` lines in a file called `.env` in the project folder become environment variables.

Why it exists: `.env.example` says "copy me to .env", which is what anyone running the project (a teammate, a new user) will do. Before
this module nothing read `.env`, so a key put there was silently ignored and the run failed with "no API key".

Rules (each one is a tested behaviour):
  * An environment variable that is ALREADY set wins -- a value typed in the terminal (`$env:GROQ_API_KEY=...`) is never overridden.
  * A blank value (`GROQ_API_KEY=`) is skipped, so the untouched template cannot blank out a real variable.
  * `#` starts a comment, on its own line or after whitespace following an unquoted value; single or double quotes are stripped;
    an optional leading `export ` is accepted.
  * Values are never printed or logged. A missing or unreadable file is not an error.
  * `SHOPSENSE_NO_DOTENV=1` turns the whole thing off (the test suite sets it so a developer's real keys cannot leak into a test run);
    `SHOPSENSE_ENV_FILE=path` points at a different file.

It runs once, when the `shopsense` package is first imported, i.e. before any module reads its settings.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def parse_env_text(text: str) -> dict[str, str]:
    """Parse the text of a .env file into {name: value}. Blank values are dropped. Pure; touches nothing."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _LINE.match(raw)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        if value[:1] in ("'", '"'):
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        if value:
            out[name] = value
    return out


def load_env_file(path: Optional[os.PathLike] = None, *, environ=None) -> list[str]:
    """Load `path` (default: `.env` next to the shopsense package) into `environ` (default os.environ) without overriding anything
    already set. Returns the NAMES it set (never the values)."""
    env = os.environ if environ is None else environ
    if env.get("SHOPSENSE_NO_DOTENV") == "1":
        return []
    target = Path(path or env.get("SHOPSENSE_ENV_FILE") or Path(__file__).resolve().parents[1] / ".env")
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError:
        return []
    loaded = []
    for name, value in parse_env_text(text).items():
        if env.get(name):                       # set and non-empty in the shell: the shell wins
            continue
        env[name] = value
        loaded.append(name)
    return loaded
