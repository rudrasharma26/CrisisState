# CrisisState Evaluation Dataset & Gold Labeling Guidelines

This directory contains the versioned, gold-standard evaluation corpus for the **CrisisState** uncertainty-aware incident intelligence system.

---

## 1. Directory Structure

```
data/evaluation/
├── README.md              # Labeling rules and specification (this document)
├── incidents.jsonl        # Gold standard incident definitions
├── reports.jsonl          # Evaluated flood reports with gold claim and incident annotations
├── claims.jsonl           # Indexed individual gold claims
└── contradictions.jsonl   # Ground-truth pairwise contradiction and superseding relations
```

---

## 2. Gold Labeling Guidelines

### A. Incident Identity
An **Incident** represents a coherent, continuous physical crisis event bounded in space and time.
- **Spatial clustering threshold**: Reports within $\le 2.0\text{ km}$ of each other referring to the same infrastructure element, road corridor, or recognized neighborhood belong to the same incident.
- **Geographically close but distinct incidents**: Infrastructure elements that operate independently (e.g., *Sector 4 Underpass* vs *Market Square* ~1.5 km away) must be labeled as distinct incidents if their water flows and blockages are functionally disjoint.
- **Temporal clustering threshold**: Reports occurring within an 8-hour activity window without an intervening documented resolution belong to the same incident.

### B. Claim Identity
A **Claim** is an atomic proposition extracted from a report regarding an entity/subject and a specific domain attribute.
- **Subject**: Canonical landmark or infrastructure name (e.g., `Main Road`, `Sector 4 Underpass`, `Riverbank Colony`).
- **Claim Type**: One of the 7 frozen vocabulary types:
  1. `ROAD_ACCESS`: `PASSABLE`, `IMPASSABLE`
  2. `WATER_LEVEL`: `LOW`, `MODERATE`, `HIGH`, `CRITICAL`
  3. `TRAFFIC_STATUS`: `MOVING`, `SLOW`, `STOPPED`
  4. `PEOPLE_TRAPPED`: `YES`, `NO`
  5. `BUILDING_DAMAGE`: `NONE`, `DAMAGED`, `COLLAPSED`
  6. `FLOODED`: `YES`, `NO`
  7. `EVACUATION`: `ORDERED`, `NOT_ORDERED`, `IN_PROGRESS`, `COMPLETED`
- **Value**: The normalized enum string.

### C. Contradiction vs. Temporal Supersession
- **Contemporaneous Contradiction (`CONTRADICTS`)**: Two claims on the same entity and attribute that assert mutually exclusive states within the active contradiction window ($\Delta t \le 2.0\text{ hours}$).
  - Example: Report at 09:00 asserts `ROAD_ACCESS = PASSABLE`, report at 09:20 asserts `ROAD_ACCESS = IMPASSABLE`. Neither can be assumed true; the state is in conflict.
- **Temporal Supersession (`SUPERSEDES`)**: Two claims on the same entity and attribute that assert differing states across an extended time gap ($\Delta t > 2.0\text{ hours}$).
  - Example: Report at 08:00 asserts `ROAD_ACCESS = IMPASSABLE`; report at 12:30 asserts `ROAD_ACCESS = PASSABLE` with water receding. The later observation supersedes the earlier one representing physical progression, not an informational clash.

### D. Modality Guidelines
Gold claims are tagged with their epistemic modality:
1. `ASSERTED`: Direct, unhedged factual claim (e.g., *"Main Road is completely impassable"*).
2. `QUALIFIED`: Conditional, partial, or vehicle-restricted claim (e.g., *"Only emergency vehicles can pass"*, *"Road is flooded only in the left lane"*).
3. `NEGATED`: Syntactically or semantically negated claim (e.g., *"Main Road is not blocked"*, *"No residents are trapped"*).
4. `AMBIGUOUS`: Hedged, speculative, or uncertain statements (e.g., *"Main Road may be blocked"*, *"Unconfirmed reports suggest water is rising"*).

### E. Test Case Categories (A through I)
The corpus contains ~80 reports systematically covering 9 evaluation categories:
- **`EXACT_LEXICAL`**: Standard phrasing directly matching Phase 1 regexes.
- **`PARAPHRASE`**: Alternative vocabulary describing the same reality (e.g., *"marooned"* instead of *"trapped"*, *"traffic halted"* instead of *"stopped"*).
- **`NEGATION`**: Statements stating absence or denial of condition.
- **`HEDGING`**: Epistemic markers (*"might"*, *"possibly"*, *"seems"*).
- **`QUALIFIED_SCOPED`**: Claims with conditions or restrictions.
- **`GEO_CLOSE_DISTINCT`**: Different incidents within close physical proximity.
- **`SEMANTIC_SIMILAR_DISTINCT`**: Similar language describing separate incidents.
- **`CONTRADICTION`**: Clashing claims within 2-hour window.
- **`TEMPORAL_SUPERSESSION`**: State updates separated by >2 hours.
