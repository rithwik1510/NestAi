# NestAi - Local Finance & Knowledge Copilot

<div align="center">
  <img src="UI DEMO.png" alt="NestAi UI" width="100%" />
  <h3>Private-by-design assistant that fuses your finances and notes without leaving your device.</h3>
  <p>
    <a href="https://img.shields.io/badge/Inference-Local-success"> 
      <img src="https://img.shields.io/badge/Inference-Local-success" alt="Local inference badge" />
    </a>
    <a href="https://img.shields.io/badge/Stack-FastAPI%20%7C%20HTMX-blueviolet">
      <img src="https://img.shields.io/badge/Stack-FastAPI%20%7C%20HTMX-blueviolet" alt="Stack badge" />
    </a>
    <a href="https://img.shields.io/badge/Ollama-qwen2.5%3A3b--instruct-informational">
      <img src="https://img.shields.io/badge/Ollama-qwen2.5%3A3b--instruct-informational" alt="Ollama badge" />
    </a>
  </p>
</div>

---

## Snapshot
| Pillar | Details |
| --- | --- |
| Local-only inference | Every request and embedding terminates at your Ollama daemon (default `http://localhost:11435`). |
| Finance-aware UX | Quick actions like "NestAi finance brief" and right-rail tips keep money + task flows cohesive. |
| Deterministic runs | Seeded generation plus persisted prompt hash, model version, and latency for replay. |
| Hybrid retrieval | BM25 + embeddings (nomic-embed-text) with cite-or-abstain synthesis; reranker flag kept off by default. |
| Production hygiene | Readiness gates, `/diagnostics` console, `make validate`, golden evals, structured logging. |

> **Everything you see in the screenshot ships in this repo**: glass chat canvas, hero pills, telemetry, typing shimmer, and diagnostics cards.

---

## Architecture
- **API**: FastAPI, Pydantic v2, PostgreSQL 16 + `pgvector`, APScheduler hooks for ingestion and maintenance.
- **Frontend**: HTMX + Tailwind with dark/light auto detect, hero copy, quick chips, and responsive composer.
- **Models**: `qwen2.5:3b-instruct` for chat, `nomic-embed-text` for embeddings (both via Ollama).
- **Retrieval**: Tantivy/Whoosh BM25 + dense vectors, optional reranker stub, cite-or-abstain policy enforced upstream.
- **Tooling**: `run.py` bootstrapper, `make validate`, `make eval`, `make diagnostics`, replay-ready run store.

---

## Getting Started
### Prerequisites
- Python 3.11+  
- Ollama with chat + embed models pulled locally  
- PostgreSQL 16 with the `pgvector` extension

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

ollama serve
ollama pull qwen2.5:3b-instruct
ollama pull nomic-embed-text
```

### Launch
```powershell
python run.py                # bootstrap, readiness check, start FastAPI
python run.py --skip-install # reuse existing environment
make dev                     # uvicorn --reload via Make
```
Open `http://localhost:8000` and chat. The composer posts `{"question": "...", "mode": "synthesize"}` to `/api/chat` and renders JSON with citations, bullets, conflicts, or abstain banner.

---

## Operations
| Command | What it does |
| --- | --- |
| `python run.py` | Creates/uses the venv, installs deps, verifies Ollama, launches uvicorn. |
| `make diagnostics` | Runs `python -m pka.app.scripts.ollama_diagnostics` for a CLI health snapshot. |
| `make validate` | Readiness probes + embedding smoke test + deterministic chat turn; writes `validation_report.json`. |
| `make eval` | Executes the golden dataset at `pka/app/services/evals/datasets/personal_golden.yaml` and writes `eval_report.md`. |
| `make dev` | Runs uvicorn with autoreload for UI iteration. |

`/diagnostics` mirrors these checks with glowing cards for probes, validation stats, and an operational checklist.

---

## Diagnostics and Evals
1. `make validate` after any model switch or hardware tweak. Archive the JSON artifact per release for traceability.  
2. `GET /health` exposes the same probes the startup gate enforces (daemon reachability, model presence, vector dim).  
3. `make eval` prints Attribution@NoAnswer metrics plus latency stats while updating `eval_report.md`.  

---

## Security Notes
- No external APIs or telemetry; inference happens via `localhost`.  
- Deletes cascade through documents, chunks, vectors, and BM25 entries, leaving an audit stub.  
- Structured logs capture model, latency, prompt hash (not raw prompt) for deterministic replay.  

---

## Roadmap Blips
- Markdown, PDF (text layer), and email ingestion pipelines with SHA-based dedupe and chunk overlap.  
- Strict-mode toggle front and center to enforce cite-or-abstain in real time.  
- Library replay viewer for diffing runs by seed/prompt version.  
- Optional local reranker flag once profiling settles.  

---

Built for privacy-heavy workflows. Ask NestAi for a "finance brief" and stay on-device the entire time.
