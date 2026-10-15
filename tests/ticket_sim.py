"""
Pure-Python executor for the capstone-M5 ticket graph, used by the test suite and dev_smoke_check.py.

The implementation now lives in shopsense/eval/offline.py (so scripts and the API's offline mode can use it too); this module
re-exports it, so every existing `from ticket_sim import ...` keeps working.
"""
from shopsense.eval.offline import (Paused, RetrievalOnlyKB, _ACCUMULATE, _approval_payload, keyword_extract, merge,  # noqa: F401
                                    resume, run)
