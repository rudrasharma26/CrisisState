# CrisisState

> **Uncertainty-Aware Incident Intelligence System for Disaster-Response Analysts**
> Primary Domain (v1): Urban Flooding

CrisisState ingests noisy, incomplete, and contradictory crisis observations and turns them into an evolving incident state with explicit evidence, uncertainty, and change history.

---

## Core Principles

- **Zero-Cost & CPU-First**: ₹0 to build and run. No paid APIs, cloud services, or proprietary models.
- **Deterministic Reasoning**: Rule-based claim extraction, explicit pairwise contradiction matrices, and transparent mathematical scoring.
- **Traceable Provenance**: Every state mutation records its originating report, previous value, new value, and reasoning snapshot.
- **Modular Monolith**: Clean domain models, SQLite persistence, and FastAPI REST endpoints without unnecessary infrastructure.

---

## Architecture & Project Structure

```
CrisisState/
├── crisisstate/
│   ├── app/
│   │   ├── main.py                # FastAPI application setup and CORS middleware
│   │   └── api/
│   │       └── routes.py          # Endpoints: /api/reports, /api/incidents, /api/replay
│   ├── domain/
│   │   ├── models.py              # Report, Claim, EvidenceLink, Incident, IncidentStateSnapshot
│   │   └── vocabulary.py          # Claim types, allowed values, contradiction pairs
│   ├── engine/
│   │   ├── extractor.py           # Deterministic claim & entity extractor
│   │   ├── matcher.py             # Spatio-temporal and entity-based incident matcher
│   │   ├── contradiction.py       # Deterministic contradiction & compatibility engine
│   │   ├── attention.py           # Transparent attention & severity scoring
│   │   └── pipeline.py            # 10-step incident intelligence processing pipeline
│   ├── storage/
│   │   ├── database.py            # SQLite connection and schema migration
│   │   └── repository.py          # Typed repository for incidents, reports, claims, snapshots
│   ├── replay/
│   │   └── runner.py              # Sequential replay runner for scenario streams
│   └── data/
│       └── synthetic_floods.json  # 9 synthetic reports with duplicates, contradictions, state updates
├── tests/
│   ├── test_acceptance_replay.py  # End-to-end scenario verification
│   ├── test_api.py                # FastAPI endpoint tests
│   ├── test_attention.py          # Attention score factors breakdown
│   ├── test_contradiction.py      # Claim incompatibility & superseding tests
│   ├── test_evidence.py           # Corroboration & contradiction evidence linking
│   ├── test_extractor.py          # Regex & vocabulary claim extraction tests
│   ├── test_matcher.py            # Spatio-temporal incident clustering tests
│   ├── test_models.py             # Domain models serialization & validation
│   ├── test_state_updates.py      # State snapshot diff tracking tests
│   └── test_storage.py            # SQLite storage CRUD tests
├── replay_demo.py                 # Single-command executable CLI demo script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Supported Claim Vocabulary & Contradiction Matrix

The system tracks 7 frozen vocabulary types in the urban flooding domain:

| Claim Type | Allowed Values | Direct Contradictions |
| :--- | :--- | :--- |
| `ROAD_ACCESS` | `PASSABLE`, `IMPASSABLE` | `PASSABLE` vs `IMPASSABLE` |
| `WATER_LEVEL` | `LOW`, `MODERATE`, `HIGH`, `CRITICAL` | `LOW`/`MODERATE` vs `CRITICAL`, `LOW` vs `HIGH` |
| `TRAFFIC_STATUS` | `MOVING`, `SLOW`, `STOPPED` | `MOVING` vs `STOPPED` |
| `PEOPLE_TRAPPED` | `YES`, `NO` | `YES` vs `NO` |
| `BUILDING_DAMAGE` | `NONE`, `DAMAGED`, `COLLAPSED` | `NONE` vs `DAMAGED`/`COLLAPSED` |
| `FLOODED` | `YES`, `NO` | `YES` vs `NO` |
| `EVACUATION` | `ORDERED`, `NOT_ORDERED`, `IN_PROGRESS`, `COMPLETED` | `NOT_ORDERED` vs `ORDERED`/`IN_PROGRESS` |

---

## 10-Step Pipeline Execution

For every incoming report:
1. **Ingest Report**: Receive unstructured report text, timestamp, coordinates, and metadata.
2. **Normalize Metadata**: Clean text and normalize timestamps to UTC.
3. **Incident Matching**: Cluster with active incidents within spatial radius and time window, or initialize a new incident.
4. **Extract Claims**: Deterministically extract entities and domain propositions.
5. **Link Claims**: Assign claims to the incident.
6. **Compare Claims**: Match against prior claims for the same entity and proposition type.
7. **Detect Contradictions**: Identify conflicting claims within the active contradiction window (or register temporal superseding).
8. **Attach Evidence**: Create explicit `EvidenceLink` records (`SUPPORTS`, `CONTRADICTS`, or `NEUTRAL`) preserving provenance.
9. **Update Incident State**: Update known state summary (`CONFLICTING` or latest value), assess severity, compute attention score factors, and record `IncidentStateSnapshot` diffs.
10. **Traceable Representation**: Return complete structured audit log and persist to SQLite.

---

## Transparent Attention Score

CrisisState does not use black-box attention weights. Attention levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) are computed from transparent contributing factors:

```json
{
  "attention_level": "CRITICAL",
  "factors": {
    "severity": 1.0,
    "conflict": 0.955,
    "recency": 0.958,
    "new_evidence": 1.0,
    "uncertainty": 0.973
  }
}
```

- **Severity**: Driven by high-risk claims (e.g., `PEOPLE_TRAPPED=YES`, `BUILDING_DAMAGE=COLLAPSED`, `WATER_LEVEL=CRITICAL`).
- **Conflict**: Ratio of active contradicted claims to total propositions.
- **Recency**: Time decay relative to the latest update.
- **New Evidence**: Volume and arrival density of observations.
- **Uncertainty**: Function of active contradiction ratio and corroboration depth.

---

## Quickstart & Verification

### 1. Requirements & Installation

```bash
pip install -r requirements.txt
```

### 2. Run the Acceptance Replay Demo

Execute the synthetic flood replay in the terminal with one command:

```bash
python replay_demo.py
```

### 3. Run the Test Suite

Execute all unit, integration, and evaluation tests with pytest:

```bash
python -m pytest tests/ -v
```

### 4. Launch the FastAPI Server

```bash
uvicorn crisisstate.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI docs will be available at `http://localhost:8000/docs`.

### API Endpoints

- `POST /api/reports`: Ingest a single report through the 10-step pipeline.
- `GET /api/reports`: List all stored reports.
- `GET /api/incidents`: List all identified incidents with current attention level and state summary.
- `GET /api/incidents/{incident_id}`: Detailed view of an incident with all linked claims and evidence.
- `GET /api/incidents/{incident_id}/timeline`: Chronological timeline of state change snapshots.
- `POST /api/replay`: Replay the synthetic stream on demand.

---

## Evaluation

CrisisState includes a dedicated, reproducible evaluation harness to measure extraction precision/recall, incident clustering, and contradiction resolution against a gold-standard benchmark corpus.

### Evaluation Corpus (`data/evaluation/`)

The evaluation dataset represents hand-crafted synthetic flood crisis reports with gold-standard incident, claim, modality, and contradiction labels. The corpus systematically covers 9 test categories:
- **`EXACT_LEXICAL`**: Standard phrasings matching rule patterns directly.
- **`PARAPHRASE`**: Semantic variations and alternative vocabularies for identical underlying claims.
- **`NEGATION`**: Assertions describing the explicit absence or denial of a condition.
- **`HEDGING`**: Epistemic uncertainty and unconfirmed statements.
- **`QUALIFIED_SCOPED`**: Claims with conditions or restrictions (e.g. emergency vehicles only).
- **`GEO_CLOSE_DISTINCT`**: Geographically proximate incidents (~400m apart) that must remain distinct.
- **`SEMANTIC_SIMILAR_DISTINCT`**: Similar language describing independent events in different locations.
- **`CONTRADICTION`**: Clashing claims within the active contradiction window ($\le 2.0\text{ h}$).
- **`TEMPORAL_SUPERSESSION`**: Sequential state updates across an evolving timeline ($> 2.0\text{ h}$).

### Metrics

1. **Claim Extraction (`Precision`, `Recall`, `F1`)**: Evaluates atomic proposition identification against gold subject and claim type pairs.
2. **Claim Normalization (`Accuracy`)**: Evaluates whether true-positive extracted claims map to the correct canonical enum values.
3. **Modality Classification (`Accuracy`)**: Measures classification of epistemic modality (`ASSERTED`, `QUALIFIED`, `NEGATED`, `AMBIGUOUS`).
4. **Incident Matching**:
   - **`False Merge Rate`**: Fraction of report pairs belonging to *different* gold incidents that were incorrectly clustered together.
   - **`Missed Merge Rate`**: Fraction of report pairs belonging to the *same* gold incident that failed to cluster together.
5. **Contradiction Detection (`Precision`, `Recall`, `F1`)**: Measures accuracy of identifying pairwise conflicting reports.
6. **Replay Determinism (`PASS` / `FAIL`)**: Ensures that running the evaluation corpus repeatedly on fresh in-memory state yields 100% identical incident partitions, extracted claims, and evidence links.
7. **Performance**: Total runtime in seconds and average per-report processing latency in milliseconds.

### Frozen Phase 1 Baseline

The current Phase 1 deterministic rule-based engine serves as the **frozen baseline** (`v1.0.0-phase1-frozen`). Future semantic iterations in Phase 2 will be compared directly against this exact baseline to ensure that semantic enhancements improve recall without degrading precision or introducing false merges.

### Running the Evaluation Harness

Execute the evaluation harness via CLI:

```bash
python -m crisisstate.evaluation.runner --dataset-dir data/evaluation --output baseline_report.json --save-baseline-config baseline_config.json
```
