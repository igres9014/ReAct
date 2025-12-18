# ReAct Demo

This repository contains a collection of small ReAct/LangGraph experiments. The `ReAct` package holds scripts for standing up agents, moving data into vector stores, and exposing the agent over HTTP.

## Project layout

- `ReAct/` – main Python package with agent scripts and utilities.
  - `ReAct_2_fixed.py` builds the core ReAct agent used by the examples.
  - `fastapi_app.py` wires the agent into a FastAPI service with `/query` and `/healthz` endpoints.
  - `chroma_*` and `docs_*` utilities demonstrate working with Chroma and CSV inputs.

## Getting started

1. Use Python 3.11+ in a virtual environment.
2. Install dependencies:
   ```bash
   cd ReAct
   pip install -r requirements.txt
   ```
3. Provide any secrets or environment variables required by your LangChain/OpenAI setup.

## Running the FastAPI example

Start the HTTP service using Uvicorn after installing dependencies:

```bash
cd ReAct
uvicorn fastapi_app:app --reload
```

The app starts the agent once during FastAPI startup, exposes `/query` for sending prompts, and `/healthz` for readiness checks. If the agent fails to initialize (for example, missing credentials), the error is returned as a 503 from `/query` and reported in the health endpoint.

## Invoking the agent directly

You can also experiment with the agent by importing `build_agent` from `ReAct_2_fixed.py` in your own scripts or notebooks. The helper returns the agent and its MCP client so you can manage the lifecycle yourself.
