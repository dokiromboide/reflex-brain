# Reflex Brain

**Continual learning brain for AI agents** — lessons, skills, and memories with evidence-based refinement, rollback, and GraphRAG retrieval.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

## Overview

Reflex Brain is an **agent-agnostic, continual learning memory system** that gives AI agents the ability to:

- **Learn continuously** from interactions — extract lessons, skills, and memories with evidence
- **Refine knowledge** through evidence-based updates (`/refine` equivalent) with full audit trail
- **Rollback safely** — snapshot-based versioning with one-click revert
- **Retrieve intelligently** — GraphRAG + continual memory hybrid queries with quality filtering
- **Run anywhere** — MCP server for Hermes/Claude Code/Codex, or as standalone library

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT LAYER                            │
│  Hermes │ Claude Code │ Codex │ OpenInterpreter │ Custom   │
└────────────────────────────┬────────────────────────────────┘
                             │ MCP / Python API
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    REFLEX BRAIN CORE                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Lessons    │  │  Skills     │  │  Memories   │          │
│  │  (versioned)│  │  (executable)│  │  (episodic) │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │  Refinement Engine   │  ← Evidence-based      │
│              │  (propose/validate/  │     updates + rollback │
│              │   apply/rollback)    │                        │
│              └──────────┬───────────┘                        │
│                         │                                    │
│              ┌──────────▼───────────┐                        │
│              │  Snapshot Store      │                        │
│              │  (full checkpoints)  │                        │
│              └──────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Lessons** | Versioned knowledge units with evidence, confidence scores, and semantic clusters |
| **Skills** | Executable capabilities (Python/JS) with interfaces, tests, and versioning |
| **Memories** | Episodic memories with importance decay and temporal context |
| **Refinement Engine** | Proposes, validates, and applies updates based on evidence weight |
| **Snapshots** | Full state checkpoints for safe rollback |
| **Hybrid Querier** | Combines GraphRAG (conversations) + Continual (lessons/skills/memories) |
| **Quality Filtering** | Penalizes tool outputs, boosts substantial content, prioritizes high-value types |
| **MCP Server** | Thin wrapper exposing `reflex_*` tools over stdio |
| **Daemon** | Background processor for passive capture + continuous embedding |

## Installation

```bash
# From PyPI (when published)
pip install reflex-brain

# From source
git clone https://github.com/jesuscaicedo800/reflex-brain.git
cd reflex-brain
pip install -e ".[dev]"
```

## Quickstart

### As MCP Server (Hermes, Claude Code, Codex)

```yaml
# config.yaml
mcp_servers:
  reflex-brain:
    command: "python"
    args: ["-m", "continual_brain.mcp.server"]
    env:
      HF_HUB_OFFLINE: "1"
      REFLEX_DB_PATH: "~/reflex-brain/continual.db"
      REFLEX_FAISS_PATH: "~/reflex-brain/"
```

Restart your agent — tools available: `reflex_query`, `reflex_propose_lesson`, `reflex_apply_refinement`, `reflex_rollback`, `reflex_snapshot`.

### As Python Library

```python
from continual_brain import ReflexBrain

brain = ReflexBrain(db_path="continual.db")

# Query (GraphRAG + Continual hybrid)
results = brain.query("DIAN facturación electrónica", top_k=5)

# Propose a lesson from recent session
proposal = brain.propose_lesson("DIAN compliance", session_id="sess_123")

# Apply with evidence threshold
brain.apply_refinement(proposal, auto_apply=True)

# Rollback if needed
brain.rollback(refinement_id="ref_abc")
```

### Run Daemon (Background Processing)

```bash
# Foreground
reflex-brain daemon --poll-interval 3 --batch-size 50

# Background service
reflex-brain daemon --daemonize
```

## Migration from GraphRAG

If you have an existing Hermes Brain graph:

```bash
reflex-brain migrate --graph-dir ~/.hermes/brain --output continual.db
```

Converts clusters + labeled nodes → lessons v1 (confidence=0.6).

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `REFLEX_DB_PATH` | `./continual.db` | SQLite database path |
| `REFLEX_FAISS_PATH` | `./` | FAISS index directory |
| `HF_HUB_OFFLINE` | `0` | Set `1` to disable HF Hub requests |
| `REFLEX_EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy continual_brain

# Pre-commit
pre-commit install
```

## Project Structure

```
reflex-brain/
├── continual_brain/
│   ├── core/           # Models, Store, RefinementEngine, Evidence
│   ├── query/          # BrainQuerier, ContinualQuerier, HybridQuerier
│   ├── daemon/         # Processor, Extractor, Embedder
│   ├── mcp/            # Thin MCP server wrapper
│   └── cli/            # CLI entry points
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── scripts/            # Migration, verification
└── pyproject.toml
```

## License

MIT © Jesus Caicedo

## Related

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) — AI agent with Brain MCP
- [Prime Intellect Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) — Inspiration for Continual Harness
- [Open Interpreter](https://github.com/openinterpreter/openinterpreter) — Agent harness for open models