"""
Example FastAPI app wiring the ReAct LangGraph agent to an HTTP endpoint.

Run with:
    uvicorn fastapi_app:app --reload

The app starts the agent once at startup and exposes a single `/query`
endpoint that proxies a prompt to the agent. Startup failures are reported
through the API so you can inspect missing environment variables or MCP
connectivity issues.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from ReAct_2_fixed import build_agent

app = FastAPI(title="ReAct FastAPI example")


class QueryRequest(BaseModel):
    """Prompt payload for the demo endpoint."""

    prompt: str


class QueryResponse(BaseModel):
    """Simplified agent reply."""

    answer: str
    raw: Optional[Dict[str, Any]] = None


_agent_state: Dict[str, Any] = {"agent": None, "mcp_client": None, "startup_error": None}


async def _close_client(client: Any) -> None:
    """Best-effort shutdown helper for the MCP client."""

    if client is None:
        return

    for name in ("aclose", "close", "shutdown", "stop", "disconnect", "terminate", "close_all"):
        fn = getattr(client, name, None)
        if not fn:
            continue
        try:
            if asyncio.iscoroutinefunction(fn):
                await fn()
            else:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            return
        except Exception:
            continue


@app.on_event("startup")
async def _startup_agent() -> None:
    """Spin up the ReAct agent once when the FastAPI app boots."""

    try:
        agent, mcp_client, _tools = await build_agent()
        _agent_state["agent"] = agent
        _agent_state["mcp_client"] = mcp_client
    except Exception as exc:  # pragma: no cover - informational only
        _agent_state["startup_error"] = str(exc)


@app.on_event("shutdown")
async def _shutdown_agent() -> None:
    await _close_client(_agent_state.get("mcp_client"))


@app.post("/query", response_model=QueryResponse)
async def query_agent(payload: QueryRequest) -> QueryResponse:
    """Send a prompt to the ReAct agent and return the response."""

    if _agent_state.get("agent") is None:
        detail = _agent_state.get("startup_error") or "Agent not ready"
        raise HTTPException(status_code=503, detail=detail)

    try:
        result = await _agent_state["agent"].ainvoke({"messages": [HumanMessage(content=payload.prompt)]})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {exc}") from exc

    answer = ""
    if isinstance(result, dict):
        answer = str(result.get("output") or result.get("content") or "")
    if not answer:
        answer = str(result)

    return QueryResponse(answer=answer, raw=result if isinstance(result, dict) else None)


@app.get("/healthz")
async def healthcheck() -> Dict[str, str]:
    status = "ready" if _agent_state.get("agent") else "initializing"
    if _agent_state.get("startup_error"):
        status = "error"
    return {"status": status}
