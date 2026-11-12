<div align="center">
  <img src="docs/trailsense-logo.svg" alt="TrailSense AI" width="360" />
  <p><strong>Ask about a trail. Tools fetch the facts, the rules engine decides, the AI explains.</strong></p>

  ![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
  ![Pydantic](https://img.shields.io/badge/Pydantic-2-E92063?logo=pydantic&logoColor=white)
  ![Ollama](https://img.shields.io/badge/Ollama-local-000000?logo=ollama&logoColor=white)
</div>

## About

A small FastAPI service where you ask about a hiking trail in plain English, like *"How hard is the Wolf Trail in Gatineau Park right now?"*. A local LLM calls my tools to get live weather, trail info and elevation, and my own rules engine decides the difficulty. The LLM only collects the data and explains the result.

> **Check official park conditions before hiking.** This app gives information, not safety guarantees.

## Features

- `GET /health` health check
- Settings come from `.env` (LLM provider, model, Ollama host)
- Auto-generated API docs at `/docs`

## Run it locally

You need [uv](https://docs.astral.sh/uv/). It installs the right Python (3.12) for this project.

```powershell
git clone https://github.com/PranavTiwari44/trailsense-ai.git
cd trailsense-ai
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload
```

Then open [localhost:8000/docs](http://localhost:8000/docs).

Run the tests with `uv run pytest`.

## To do

- Data services: Open-Meteo weather and elevation, OpenStreetMap trails (with a local fallback)
- Difficulty rules engine + `GET /difficulty`
- LLM tool calling with Ollama, behind a swappable provider interface
- `POST /explain` with structured JSON output
- Agent tests with a fake LLM client, and an eval set
