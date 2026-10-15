"""ShopSense: customer care & order operations assistant (capstone)."""
import sys as _sys

from .envfile import load_env_file as _load_env_file

_load_env_file()            # reads ./.env if present; never overrides a variable already set in the shell (see envfile.py)


def _make_output_safe() -> None:
    """A character the console encoding cannot show (e.g. a non-breaking hyphen on a Windows code page, when output is piped) must print
    as '?', never crash the script that is printing it."""
    for stream in (_sys.stdout, _sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:                      # noqa: BLE001 -- not a text stream that supports it (e.g. a test capture): leave it alone
            pass


_make_output_safe()
