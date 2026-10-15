"""
ShopSense demo API (Milestone 8): `POST /chat` and `GET /health`, a thin FastAPI wrapper over `service.ChatService`.

    pip install fastapi uvicorn
    $env:SHOPSENSE_OFFLINE = "1"                        # demo with no model key (omit for the real model + LangGraph)
    python -m uvicorn shopsense.api:app --port 8000
    python demo_api_client.py                           # or Postman: POST http://127.0.0.1:8000/chat  {"message": "...", "customer_id": "CUST-0077"}
    # interactive docs: http://127.0.0.1:8000/docs

All behaviour lives in service.py / guardrails.py (tested without FastAPI). This file only declares the two routes, the request and
response contracts, and the mapping of failures to HTTP status codes:
    422  the request itself is invalid (empty / over-long message) -- FastAPI/pydantic does this before our code runs
    503  the service cannot start (live mode without langgraph / a model key)
    500  anything unexpected: a generic body, never an internal message; the trace holds the detail
A guardrail BLOCK is not an error: the customer gets a safe holding reply, `guardrail.blocked` is true and the flags are visible.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .guardrails import MAX_MESSAGE_CHARS
from .observability.report import DEFAULT_TRACE_FILE
from .observability.tracing import set_tracer, tracer_from_env
from .service import ChatService


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    customer_id: Optional[str] = Field(default=None, max_length=40, description="DEMO ONLY: production derives this from an authenticated session")
    session_id: Optional[str] = Field(default=None, max_length=80, description="groups messages for the repeat-contact rule")


class GuardrailInfo(BaseModel):
    passed: bool
    blocked: bool
    flags: List[str]


class ChatResponse(BaseModel):
    ticket_id: str
    session_id: str
    outcome: str                      # resolved | escalated | awaiting_approval
    reply: str
    awaiting_approval: bool
    guardrail: GuardrailInfo
    degraded: bool
    input_flags: List[str]
    # deliberately NOT returned: escalation codes, routing reasons, order facts, the policy answer, the trace -- internal fields,
    # and some of them (an abuse-flag code) must never reach a customer.


def create_app(service: Optional[ChatService] = None) -> FastAPI:
    svc = service or ChatService()
    tracer = tracer_from_env(trace_file=DEFAULT_TRACE_FILE)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        set_tracer(tracer)
        yield
        tracer.flush()                # push buffered LangFuse events before the process exits

    app = FastAPI(title="ShopSense Customer Care", version="1.0", lifespan=lifespan)
    app.state.service = svc

    @app.get("/health")
    def health() -> dict:
        return svc.health()

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        try:
            return ChatResponse(**svc.chat(req.message, customer_id=req.customer_id, session_id=req.session_id))
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception:                                  # noqa: BLE001 -- never leak internals to a customer
            raise HTTPException(status_code=500, detail="internal error; the request was not completed")

    return app


app = create_app()
