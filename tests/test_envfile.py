"""
.env support, the .env.example template and the ask.py command-line entry point -- the things a new user touches first.
"""
import contextlib
import io
import os
import re
from pathlib import Path

import ask
from shopsense import envfile

ROOT = Path(__file__).resolve().parents[1]


def test_parse_handles_comments_quotes_export_and_blank_values():
    text = '\n'.join([
        "# a comment", "", "GROQ_API_KEY=abc123   # trailing comment", 'SHOPSENSE_LLM_MODEL="groq/openai/gpt-oss-20b"',
        "export SHOPSENSE_OFFLINE='1'", "EMPTY=", "SPACED = value with spaces", "not a line", "TOKEN=abc#notacomment",
    ])
    got = envfile.parse_env_text(text)
    assert got == {"GROQ_API_KEY": "abc123", "SHOPSENSE_LLM_MODEL": "groq/openai/gpt-oss-20b", "SHOPSENSE_OFFLINE": "1",
                   "SPACED": "value with spaces", "TOKEN": "abc#notacomment"}


def test_the_shell_wins_and_blank_shell_values_are_filled(tmp_path):
    f = tmp_path / ".env"
    f.write_text("A=from_file\nB=from_file\nC=from_file\n")
    env = {"A": "from_shell", "B": ""}                    # A is set; B is set but empty; C is unset
    loaded = envfile.load_env_file(f, environ=env)
    assert env == {"A": "from_shell", "B": "from_file", "C": "from_file"} and sorted(loaded) == ["B", "C"]


def test_missing_file_is_not_an_error_and_the_off_switch_works(tmp_path):
    assert envfile.load_env_file(tmp_path / "nope.env", environ={}) == []
    f = tmp_path / ".env"
    f.write_text("A=1\n")
    env = {"SHOPSENSE_NO_DOTENV": "1"}
    assert envfile.load_env_file(f, environ=env) == [] and "A" not in env


def test_the_untouched_template_sets_nothing():
    """Copying .env.example to .env and changing nothing must not blank out or invent any setting."""
    assert envfile.parse_env_text((ROOT / ".env.example").read_text(encoding="utf-8")) == {}


def test_every_setting_the_code_reads_is_documented_in_the_template():
    template = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", template, re.M))
    used = set()
    for path in list((ROOT / "shopsense").rglob("*.py")) + list(ROOT.glob("*.py")):
        used |= set(re.findall(r"""["']((?:SHOPSENSE|LANGFUSE)_[A-Z0-9_]+|GROQ_API_KEY|GEMINI_API_KEY)["']""", path.read_text(encoding="utf-8")))
    missing = sorted(used - documented)
    assert not missing, f"read by the code but missing from .env.example: {missing}"


def test_env_is_ignored_by_git_and_the_template_is_not():
    ignore = (ROOT / ".gitignore").read_text()
    assert re.search(r"^\.env$", ignore, re.M) and ".env.example" not in ignore


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    keep = os.environ.get("SHOPSENSE_OFFLINE")
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ask.main(argv)
    finally:
        if keep is None:
            os.environ.pop("SHOPSENSE_OFFLINE", None)
        else:
            os.environ["SHOPSENSE_OFFLINE"] = keep
    return code, out.getvalue(), err.getvalue()


def test_ask_examples_lists_every_example_command():
    code, out, _ = _run(["--examples"])
    assert code == 0 and out.count("python ask.py") == len(ask.EXAMPLES)


def test_ask_answers_offline_with_the_real_routing():
    code, out, _ = _run(["--offline", "I want my INR 25,516 back for ORD-4553 RIGHT NOW. The smartwatch and earbuds are junk.", "--customer", "CUST-0142"])
    assert code == 0 and "outcome: awaiting_approval" in out and "needs approval: yes" in out
    code, out, _ = _run(["--offline", "--json", "Where is my order ORD-4445? Any tracking update?", "--customer", "CUST-0077"])
    assert code == 0 and '"outcome": "resolved"' in out and "escalation" not in out.lower()


def test_ask_without_a_model_key_says_what_to_do_instead_of_a_traceback(monkeypatch):
    for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SHOPSENSE_OFFLINE"):
        monkeypatch.delenv(k, raising=False)
    code, _, err = _run(["hello"])
    assert code == 2 and ".env" in err and "--offline" in err
