"""SQLite connection and schema manager for CrisisState.

M1 adds:
  - New columns to `claims` (extraction_method, extraction_confidence,
    matched_exemplar_id, similarity_score, source_span, value_modality,
    qualifier, candidate_values)
  - New columns to `evidence_links` (strength_components, strength_total,
    independence_group_id)
  - New column to `incidents` (claim_state_entries)
  - New tables: claim_exemplars, unresolved_spans, incident_matches

Migration strategy: all new columns have DEFAULT values so that
``ALTER TABLE … ADD COLUMN`` succeeds on existing databases without
destroying any existing data.  New tables use ``CREATE TABLE IF NOT EXISTS``.
"""

import sqlite3
from typing import Optional


# ── Base schema (Phase 1 tables — never altered structurally) ────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    location_name TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    current_severity TEXT NOT NULL,
    current_attention_level TEXT NOT NULL,
    attention_factors TEXT NOT NULL,
    current_confidence REAL NOT NULL,
    linked_report_ids TEXT NOT NULL,
    claim_ids TEXT NOT NULL,
    state_summary TEXT NOT NULL,
    -- M1: structured per-(incident, claim_type) state (JSON list, default empty)
    claim_state_entries TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    incident_id TEXT,
    report_id TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    value TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    supporting_evidence TEXT NOT NULL,
    contradicting_evidence TEXT NOT NULL,
    -- M1: new semantic-layer preparation columns
    extraction_method TEXT NOT NULL DEFAULT 'LEXICAL',
    extraction_confidence REAL,
    matched_exemplar_id TEXT,
    similarity_score REAL,
    source_span TEXT,
    value_modality TEXT NOT NULL DEFAULT 'ASSERTED',
    qualifier TEXT,
    candidate_values TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS evidence_links (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    claim_id TEXT NOT NULL,
    incident_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    -- M1: evidence strength preparation columns
    strength_components TEXT NOT NULL DEFAULT '{}',
    strength_total REAL,
    independence_group_id TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY,
    incident_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    changed_field TEXT NOT NULL,
    previous_value TEXT,
    new_value TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_reference TEXT
);

-- M1: New tables (not yet wired into Phase 1 pipeline)

CREATE TABLE IF NOT EXISTS claim_exemplars (
    id TEXT PRIMARY KEY,
    claim_type TEXT NOT NULL,
    value TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'HAND_AUTHORED',
    version TEXT NOT NULL DEFAULT 'v1',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unresolved_spans (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    span_text TEXT NOT NULL,
    top_candidates TEXT NOT NULL DEFAULT '[]',
    rejection_reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incident_matches (
    id TEXT PRIMARY KEY,
    report_id TEXT NOT NULL,
    incident_id TEXT NOT NULL,
    match_reason TEXT NOT NULL,
    gate_results TEXT NOT NULL DEFAULT '{}',
    semantic_score REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_incident ON claims(incident_id);
CREATE INDEX IF NOT EXISTS idx_claims_subject ON claims(subject);
CREATE INDEX IF NOT EXISTS idx_evidence_incident ON evidence_links(incident_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_incident ON snapshots(incident_id);
CREATE INDEX IF NOT EXISTS idx_unresolved_spans_report ON unresolved_spans(report_id);
CREATE INDEX IF NOT EXISTS idx_incident_matches_report ON incident_matches(report_id);
CREATE INDEX IF NOT EXISTS idx_exemplars_type_value ON claim_exemplars(claim_type, value);

-- M2: embedding cache (key = SHA256 of model_id + "||" + normalized_text)
CREATE TABLE IF NOT EXISTS embedding_cache (
    key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    vector TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model_id);
"""

# ── Migration: new columns added to existing Phase 1 tables ──────────────────
# Each statement is safe to run repeatedly (errors on duplicate column are
# caught and suppressed).

_MIGRATION_STATEMENTS = [
    # incidents
    "ALTER TABLE incidents ADD COLUMN claim_state_entries TEXT NOT NULL DEFAULT '[]'",

    # claims — M1 columns
    "ALTER TABLE claims ADD COLUMN extraction_method TEXT NOT NULL DEFAULT 'LEXICAL'",
    "ALTER TABLE claims ADD COLUMN extraction_confidence REAL",
    "ALTER TABLE claims ADD COLUMN matched_exemplar_id TEXT",
    "ALTER TABLE claims ADD COLUMN similarity_score REAL",
    "ALTER TABLE claims ADD COLUMN source_span TEXT",
    "ALTER TABLE claims ADD COLUMN value_modality TEXT NOT NULL DEFAULT 'ASSERTED'",
    "ALTER TABLE claims ADD COLUMN qualifier TEXT",
    "ALTER TABLE claims ADD COLUMN candidate_values TEXT NOT NULL DEFAULT '[]'",

    # evidence_links — M1 columns
    "ALTER TABLE evidence_links ADD COLUMN strength_components TEXT NOT NULL DEFAULT '{}'",
    "ALTER TABLE evidence_links ADD COLUMN strength_total REAL",
    "ALTER TABLE evidence_links ADD COLUMN independence_group_id TEXT",

    # M2: embedding cache table (CREATE IF NOT EXISTS is safe; listed here for
    # explicit migration record-keeping alongside the ALTER TABLE statements)
    # The table is created by SCHEMA_SQL above, so no ALTER TABLE needed here.
]


class Database:
    """Manages SQLite database connection and schema initialisation."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False allows FastAPI/threads to share connection safely
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for file-based DB
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
        return self._conn

    def init_db(self) -> None:
        conn = self.get_connection()
        with conn:
            conn.executescript(SCHEMA_SQL)
        # Run migrations for existing databases (idempotent)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Apply M1 column additions to existing Phase 1 databases.

        ``ALTER TABLE … ADD COLUMN`` raises ``OperationalError`` if the
        column already exists (e.g. fresh DB created from the new schema).
        We catch and ignore those errors so migrations are safe to re-run.
        """
        conn = self.get_connection()
        for stmt in _MIGRATION_STATEMENTS:
            try:
                with conn:
                    conn.execute(stmt)
            except sqlite3.OperationalError:
                # Column already exists — safe to ignore
                pass

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
