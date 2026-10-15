"""
Milestone 1 tests: LLM client (provider-agnostic + session state) and the
structured intake pipeline (envelope validation/repair + LLM extraction/
repair loop), all exercised against the scripted fake LLM in conftest.py.
"""
import json

import pytest
from pydantic import ValidationError

from shopsense.core.llm_client import SessionStore, llm_chat
from shopsense.core.schemas import SupportTicket, TicketExtraction
from shopsense.core.ticket_intake import (
    extract_ticket_info,
    parse_envelopes_jsonl,
    process_ticket,
    repair_envelope,
)

VALID_EXTRACTION = {
    "intent": "TRACK_DELIVERY",
    "category": "DELIVERY",
    "order_id": "ORD-4001",
    "sentiment": "neutral",
    "urgency": "low",
    "requires_human_review": False,
    "contains_injection_attempt": False,
}


# --------------------------------------------------------------------------
# llm_client: provider-agnostic client + session state
# --------------------------------------------------------------------------

def test_llm_chat_returns_reply_text(scripted_litellm):
    scripted_litellm.queue_text("Hello there!")
    reply = llm_chat("hi", model="groq/llama-3.1-8b-instant")
    assert reply == "Hello there!"
    assert scripted_litellm.calls[0]["model"] == "groq/llama-3.1-8b-instant"


def test_session_store_keeps_independent_histories(scripted_litellm):
    scripted_litellm.queue_text("Got it, Alex!")
    scripted_litellm.queue_text("Got it, Priya!")
    scripted_litellm.queue_text("Your name is Alex.")

    store = SessionStore()
    store.chat("user-alex", "My name is Alex.")
    store.chat("user-priya", "My name is Priya.")
    reply = store.chat("user-alex", "What is my name?")

    assert reply == "Your name is Alex."
    # system + 2 user + 2 assistant turns for user-alex
    assert len(store.get_or_create("user-alex")) == 5
    assert len(store.get_or_create("user-priya")) == 3


def test_session_store_reset(scripted_litellm):
    scripted_litellm.queue_text("hi")
    store = SessionStore()
    store.chat("s1", "hello")
    assert "s1" in store.active_sessions()
    store.reset("s1")
    assert "s1" not in store.active_sessions()


# --------------------------------------------------------------------------
# Envelope validation + deterministic repair
# --------------------------------------------------------------------------

def test_valid_envelope_parses():
    ticket = SupportTicket(
        ticket_id="TCK-0001", channel="web_form",
        created_at="2026-10-01T09:00:00", text="Help with my order",
    )
    assert ticket.ticket_id == "TCK-0001"


def test_envelope_rejects_bad_ticket_id():
    with pytest.raises(ValidationError):
        SupportTicket(ticket_id="BAD-0001", channel="web_form",
                      created_at="2026-10-01T09:00:00", text="Help")


def test_repair_envelope_fills_missing_fields():
    ticket, repairs = repair_envelope({"text": "Help with my order"})
    assert ticket is not None
    assert ticket.ticket_id == "TCK-0000"
    assert ticket.channel == "web_form"
    assert "generated_missing_ticket_id" in repairs
    assert "default_channel_to_web_form" in repairs
    assert "generated_created_at_timestamp" in repairs


def test_parse_envelopes_jsonl(tmp_path):
    f = tmp_path / "tickets.jsonl"
    f.write_text(
        json.dumps({"ticket_id": "TCK-0001", "channel": "email",
                    "created_at": "2026-10-01T09:00:00", "text": "Help"}) + "\n"
        + json.dumps({"text": "Missing everything else"}) + "\n"
    )
    valid, failed = parse_envelopes_jsonl(str(f))
    assert len(valid) == 2  # second one repairs cleanly
    assert len(failed) == 0


# --------------------------------------------------------------------------
# LLM-driven extraction + repair loop -- the actual M1 AI step
# --------------------------------------------------------------------------

def test_extract_ticket_info_success_first_try(scripted_litellm):
    scripted_litellm.queue_text(json.dumps(VALID_EXTRACTION))
    extraction, errors = extract_ticket_info("Where is my order ORD-4001?")
    assert errors == []
    assert isinstance(extraction, TicketExtraction)
    assert extraction.intent == "TRACK_DELIVERY"
    assert extraction.order_id == "ORD-4001"
    assert len(scripted_litellm.calls) == 1  # no repair call needed


def test_extract_ticket_info_repairs_after_bad_json(scripted_litellm):
    # First attempt: model returns invalid JSON (missing required fields).
    scripted_litellm.queue_text(json.dumps({"intent": "TRACK_DELIVERY"}))
    # Repair attempt: model returns a corrected, fully valid object.
    scripted_litellm.queue_text(json.dumps(VALID_EXTRACTION))

    extraction, errors = extract_ticket_info("Where is my order ORD-4001?", max_retries=2)

    assert extraction is not None
    assert extraction.intent == "TRACK_DELIVERY"
    assert len(errors) == 1  # one validation failure recorded before the repair succeeded
    assert len(scripted_litellm.calls) == 2  # initial extraction + one repair call


def test_extract_ticket_info_gives_up_after_max_retries(scripted_litellm):
    # Model never produces valid JSON, across every attempt.
    for _ in range(4):
        scripted_litellm.queue_text("not json at all")

    extraction, errors = extract_ticket_info("some text", max_retries=2)

    assert extraction is None
    assert len(errors) == 3  # initial + 2 retries, all failed
    assert len(scripted_litellm.calls) == 3  # initial + 2 repair calls, then gives up


def test_process_ticket_end_to_end(scripted_litellm):
    scripted_litellm.queue_text(json.dumps(VALID_EXTRACTION))
    raw = {"ticket_id": "TCK-0001", "channel": "web_form",
           "created_at": "2026-10-01T09:00:00", "text": "Where is ORD-4001?"}
    result = process_ticket(raw)
    assert result["ticket"].ticket_id == "TCK-0001"
    assert result["extraction"].intent == "TRACK_DELIVERY"
    assert result["envelope_repairs"] == []
    assert result["extraction_errors"] == []


def test_parse_retry_wait_understands_every_provider_format():
    from shopsense.core.llm_client import parse_retry_wait

    assert parse_retry_wait("Please try again in 45.2s.") == 45.2
    assert abs(parse_retry_wait("Please try again in 2m33.36s. Need more") - 153.36) < 1e-6
    assert parse_retry_wait("try again in 1h2m3s") == 3723
    assert parse_retry_wait("try again in 850ms") == 0.85
    assert parse_retry_wait("rate limited") is None
