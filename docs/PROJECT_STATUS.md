# CrisisState — Project Status

## Current State

- Phase 1: Complete
- M0 Evaluation Harness: Complete
- M1 Schema Extensions: Complete
- M2 Embedding Infrastructure: Next
- M3 Claim Exemplar Bank: Planned
- Tests: 74/74 passing

## Frozen Baseline

Phase 1 baseline:
`v1.0.0-phase1-frozen`

## Architecture

- Python
- FastAPI
- SQLite
- Pydantic
- NumPy
- scikit-learn
- sentence-transformers (planned)
- spaCy/lightweight NLP (planned)
- Streamlit (planned)

## Product

CrisisState is an uncertainty-aware incident intelligence system for conflicting
urban flood reports.

Core pipeline:

Report → Claim → Evidence → Conflict/Support → Incident State → State Evolution

## Constraints

- ₹0 development and deployment
- Local/open-source tooling
- CPU-first
- Modular monolith
- No live social-media dependency
- Public demo via Streamlit Community Cloud