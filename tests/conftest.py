"""
Shared test fixtures.

Installs a fake `litellm` module into sys.modules BEFORE any shopsense.*
module is imported, so `from litellm import completion` throughout the
codebase binds to a scriptable test double instead of the real package.
This lets the whole control flow (repair loops, tool-calling loop, ReAct
parsing) be verified deterministically, with no network call, no API key,
and no dependency on litellm actually being pip-installed in this
environment. Live runs (run_pipeline.py, ask.py) are unaffected --
conftest.py is only ever loaded by pytest.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

# A developer's real .env (API keys) must never leak into a test run: shopsense/envfile.py honours this switch at import time.
os.environ["SHOPSENSE_NO_DOTENV"] = "1"

# Tests must never write customer memory into the project's data/ folder (it would make results depend on earlier runs, and on a
# real machine would leave files behind). Every MemoryStore() built without an explicit folder now lands in a throwaway temp dir.
os.environ.setdefault("SHOPSENSE_MEMORY_DIR", tempfile.mkdtemp(prefix="shopsense_test_memory_"))

# Make the `shopsense` package importable regardless of how pytest is invoked
# (rootdir, -k from a subdirectory, etc.).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class FakeFunctionCall:
    def __init__(self, name: str, arguments):
        self.name = name
        self.arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments):
        self.id = call_id
        self.function = FakeFunctionCall(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage):
        self.message = message


class FakeResponse:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [FakeChoice(FakeMessage(content=content, tool_calls=tool_calls))]


class ScriptedLiteLLM:
    """Fake `litellm.completion` backed by a FIFO queue of canned responses
    the test sets up ahead of time, so each test controls exactly what the
    "model" says without any real LLM call."""

    def __init__(self):
        self._queue: list[FakeResponse] = []
        self.calls: list[dict] = []

    def reset(self) -> None:
        self._queue.clear()
        self.calls.clear()
        self.responder = None

    def queue_text(self, text: str) -> None:
        self._queue.append(FakeResponse(content=text))

    def queue_tool_call(self, call_id: str, name: str, arguments) -> None:
        self._queue.append(FakeResponse(content=None, tool_calls=[FakeToolCall(call_id, name, arguments)]))

    def completion(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if getattr(self, "responder", None) is not None:  # keyed fake, used by the team simulator
            return FakeResponse(content=self.responder(messages))
        if not self._queue:
            raise AssertionError("ScriptedLiteLLM: no more scripted responses queued for this call")
        return self._queue.pop(0)


def _install_fake_litellm() -> ScriptedLiteLLM:
    fake_module = types.ModuleType("litellm")
    scripted = ScriptedLiteLLM()
    fake_module.completion = scripted.completion  # type: ignore[attr-defined]
    sys.modules["litellm"] = fake_module
    return scripted


# Runs once at conftest collection time -- before pytest imports any test
# module, and therefore before any test module can import shopsense.core.
_SCRIPTED = _install_fake_litellm()


@pytest.fixture
def scripted_litellm() -> ScriptedLiteLLM:
    _SCRIPTED.reset()
    return _SCRIPTED
