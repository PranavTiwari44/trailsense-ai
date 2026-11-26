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

- `POST /explain`: ask a question in plain English, get a structured JSON answer (level, score, summary, reasons, warnings, gear tips, data time). The AI writes the words; **the rules engine decides the level and score**, and overrides the AI if they disagree
- **Structured output**: the AI must reply with JSON matching a Pydantic schema (Ollama's `format` option), validated and retried once if invalid. If the AI forgets to ask for the difficulty score, the app asks for it itself
- `GET /health` health check, auto-generated API docs at `/docs`
- **Difficulty rules engine**: scores a trail 0–100 from its length, climb, OSM difficulty tag and the next hours of weather (rain, ice, wind, heat). A thunderstorm makes it *Dangerous* whatever the score, and missing data gives *Unknown*, never *Easy*
- `GET /difficulty?trail_id=wolf-trail`: the engine's answer for a trail right now (no AI)
- **LLM tool calling**: the model asks for `find_trail`, `get_weather`, `get_elevation_gain` and `get_difficulty_score` and chains them on its own (e.g. finds the trail, then uses its id). Tool errors go back to the model as messages, never crashes, and a step limit stops runaway loops
- **Swappable AI interface**: the app talks to one `LLMClient` shape; `LLM_PROVIDER` picks the model behind it. Runs locally on Ollama (`qwen3.5:9b`) for free
- **Trail search** in OpenStreetMap: groups a trail's many path segments into one trail, adds up the length, and keeps the hardest difficulty tag
- **Local fallback** of real Gatineau Park trails, used automatically when Overpass is slow or down
- **Live weather** for the trailhead: now, the past 6 hours (for ice) and the next 6 hours
- **Elevation gain** along the trail, with requests batched and run in parallel
- Settings come from `.env` (LLM provider, model, API URLs, timeouts)

## Run it locally

You need [uv](https://docs.astral.sh/uv/). It installs the right Python (3.12) for this project.

```powershell
git clone https://github.com/PranavTiwari44/trailsense-ai.git
cd trailsense-ai
cp .env.example .env
uv sync
uv run uvicorn app.main:app --reload
```

Then open [localhost:8000/docs](http://localhost:8000/docs) and try `POST /explain` with `{"question": "How hard is the Wolf Trail in Gatineau Park right now?"}` (needs Ollama running; takes 5–60 s).

Run the tests with `uv run pytest`.

Try the data services against the real APIs:

```powershell
uv run python -m scripts.try_services "Wolf Trail"
```

Ask the local model a question and watch it use the tools (needs Ollama running):

```powershell
uv run python -m scripts.try_agent "How hard is the Wolf Trail in Gatineau Park right now?"
```

## Data

| Source | Used for | Key needed |
|---|---|---|
| [Open-Meteo](https://open-meteo.com) | Weather forecast and elevation | No |
| [OpenStreetMap](https://www.openstreetmap.org) via [Overpass](https://overpass-api.de) | Trail names, paths and difficulty tags | No |

`app/data/trails_fallback.json` is a snapshot of OpenStreetMap data for a few Gatineau Park trails (Wolf Trail, King Mountain, Luskville Falls, Pink Lake). Rebuild it with `uv run python -m scripts.build_fallback`.

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), available under the Open Database License.

## To do

- More agent tests with a fake LLM client (retry, safety net), and an eval set
- A web page for asking questions (TypeScript)
