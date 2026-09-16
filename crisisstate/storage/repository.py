"""SQLite repository implementation for CrisisState domain entities.

M1 changes:
  - save_claim / _row_to_claim handle the eight new Claim fields.
  - save_evidence_link / list_evidence_for_incident handle the three new
    EvidenceLink fields.
  - save_incident / _row_to_incident handle the new claim_state_entries field.
  - New CRUD methods for ClaimExemplar, UnresolvedSpan, and IncidentMatch.

All Phase 1 method signatures are unchanged.  New fields default to empty /
None so that callers that do not supply them continue to work correctly.
"""

import json
from datetime import datetime
from typing import List, Optional
import sqlite3

from crisisstate.domain.models import (
    Claim,
    ClaimExemplar,
    ClaimStateEntry,
    EvidenceLink,
    Incident,
    IncidentMatch,
    IncidentStateSnapshot,
    Report,
    UnresolvedSpan,
)
from crisisstate.domain.vocabulary import (
    AttentionLevel,
    ClaimStatus,
    ClaimType,
    EvidenceType,
    ExemplarSource,
    ExtractionMethod,
    MatchReason,
    SeverityLevel,
    ValueModality,
)
from crisisstate.storage.database import Database


from crisisstate.domain.models import ensure_utc


def _dt_to_iso(dt: datetime) -> str:
    return ensure_utc(dt).isoformat()


def _iso_to_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return ensure_utc(dt)


class CrisisRepository:
    """Repository providing typed storage and retrieval for CrisisState entities."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------ Reports ------------------

    def save_report(self, report: Report) -> Report:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reports 
                (id, text, timestamp, source_type, source_id, latitude, longitude, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    report.text,
                    _dt_to_iso(report.timestamp),
                    report.source_type,
                    report.source_id,
                    report.latitude,
                    report.longitude,
                    json.dumps(report.metadata),
                ),
            )
        return report

    def get_report(self, report_id: str) -> Optional[Report]:
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return Report(
            id=row["id"],
            text=row["text"],
            timestamp=_iso_to_dt(row["timestamp"]),
            source_type=row["source_type"],
            source_id=row["source_id"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            metadata=json.loads(row["metadata"]),
        )

    def list_reports(self) -> List[Report]:
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT * FROM reports ORDER BY timestamp ASC")
        rows = cursor.fetchall()
        return [
            Report(
                id=r["id"],
                text=r["text"],
                timestamp=_iso_to_dt(r["timestamp"]),
                source_type=r["source_type"],
                source_id=r["source_id"],
                latitude=r["latitude"],
                longitude=r["longitude"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    # ------------------ Incidents ------------------

    def save_incident(self, incident: Incident) -> Incident:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO incidents
                (id, title, location_name, latitude, longitude, created_at, updated_at,
                 current_severity, current_attention_level, attention_factors,
                 current_confidence, linked_report_ids, claim_ids, state_summary,
                 claim_state_entries)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident.id,
                    incident.title,
                    incident.location_name,
                    incident.latitude,
                    incident.longitude,
                    _dt_to_iso(incident.created_at),
                    _dt_to_iso(incident.updated_at),
                    incident.current_severity.value,
                    incident.current_attention_level.value,
                    json.dumps(incident.attention_factors),
                    incident.current_confidence,
                    json.dumps(incident.linked_report_ids),
                    json.dumps(incident.claim_ids),
                    json.dumps(incident.state_summary),
                    json.dumps(incident.claim_state_entries),
                ),
            )
        return incident

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_incident(row)

    def list_incidents(self) -> List[Incident]:
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT * FROM incidents ORDER BY updated_at DESC")
        return [self._row_to_incident(r) for r in cursor.fetchall()]

    def _row_to_incident(self, row: sqlite3.Row) -> Incident:
        keys = row.keys()
        claim_state_entries = (
            json.loads(row["claim_state_entries"])
            if "claim_state_entries" in keys and row["claim_state_entries"] is not None
            else []
        )
        return Incident(
            id=row["id"],
            title=row["title"],
            location_name=row["location_name"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            created_at=_iso_to_dt(row["created_at"]),
            updated_at=_iso_to_dt(row["updated_at"]),
            current_severity=SeverityLevel(row["current_severity"]),
            current_attention_level=AttentionLevel(row["current_attention_level"]),
            attention_factors=json.loads(row["attention_factors"]),
            current_confidence=row["current_confidence"],
            linked_report_ids=json.loads(row["linked_report_ids"]),
            claim_ids=json.loads(row["claim_ids"]),
            state_summary=json.loads(row["state_summary"]),
            claim_state_entries=claim_state_entries,
        )

    # ------------------ Claims ------------------

    def save_claim(self, claim: Claim) -> Claim:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO claims
                (id, incident_id, report_id, claim_type, subject, value, timestamp, status,
                 confidence, supporting_evidence, contradicting_evidence,
                 extraction_method, extraction_confidence, matched_exemplar_id,
                 similarity_score, source_span, value_modality, qualifier, candidate_values)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim.id,
                    claim.incident_id,
                    claim.report_id,
                    claim.claim_type.value,
                    claim.subject,
                    claim.value,
                    _dt_to_iso(claim.timestamp),
                    claim.status.value,
                    claim.confidence,
                    json.dumps(claim.supporting_evidence),
                    json.dumps(claim.contradicting_evidence),
                    # M1 fields
                    claim.extraction_method.value,
                    claim.extraction_confidence,
                    claim.matched_exemplar_id,
                    claim.similarity_score,
                    json.dumps(claim.source_span) if claim.source_span is not None else None,
                    claim.value_modality.value,
                    json.dumps(claim.qualifier) if claim.qualifier is not None else None,
                    json.dumps(claim.candidate_values),
                ),
            )
        return claim

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        conn = self.db.get_connection()
        cursor = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return self._row_to_claim(row)

    def list_claims_for_incident(self, incident_id: str) -> List[Claim]:
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT * FROM claims WHERE incident_id = ? ORDER BY timestamp ASC",
            (incident_id,),
        )
        return [self._row_to_claim(r) for r in cursor.fetchall()]

    def _row_to_claim(self, row: sqlite3.Row) -> Claim:
        keys = row.keys()

        def _get(col, default=None):
            return row[col] if col in keys and row[col] is not None else default

        return Claim(
            id=row["id"],
            incident_id=row["incident_id"],
            report_id=row["report_id"],
            claim_type=ClaimType(row["claim_type"]),
            subject=row["subject"],
            value=row["value"],
            timestamp=_iso_to_dt(row["timestamp"]),
            status=ClaimStatus(row["status"]),
            confidence=row["confidence"],
            supporting_evidence=json.loads(row["supporting_evidence"]),
            contradicting_evidence=json.loads(row["contradicting_evidence"]),
            # M1 fields — fall back to defaults for legacy rows
            extraction_method=ExtractionMethod(_get("extraction_method", "LEXICAL")),
            extraction_confidence=_get("extraction_confidence"),
            matched_exemplar_id=_get("matched_exemplar_id"),
            similarity_score=_get("similarity_score"),
            source_span=json.loads(_get("source_span", "null")),
            value_modality=ValueModality(_get("value_modality", "ASSERTED")),
            qualifier=json.loads(_get("qualifier", "null")),
            candidate_values=json.loads(_get("candidate_values", "[]")),
        )

    # ------------------ Evidence Links ------------------

    def save_evidence_link(self, link: EvidenceLink) -> EvidenceLink:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO evidence_links
                (id, report_id, claim_id, incident_id, link_type, confidence, reason, created_at,
                 strength_components, strength_total, independence_group_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link.id,
                    link.report_id,
                    link.claim_id,
                    link.incident_id,
                    link.link_type.value,
                    link.confidence,
                    link.reason,
                    _dt_to_iso(link.created_at),
                    # M1 fields
                    json.dumps(link.strength_components),
                    link.strength_total,
                    link.independence_group_id,
                ),
            )
        return link

    def list_evidence_for_incident(self, incident_id: str) -> List[EvidenceLink]:
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT * FROM evidence_links WHERE incident_id = ? ORDER BY created_at ASC",
            (incident_id,),
        )
        results = []
        for r in cursor.fetchall():
            keys = r.keys()

            def _get(col, default=None, _r=r, _keys=keys):
                return _r[col] if col in _keys and _r[col] is not None else default

            results.append(
                EvidenceLink(
                    id=r["id"],
                    report_id=r["report_id"],
                    claim_id=r["claim_id"],
                    incident_id=r["incident_id"],
                    link_type=EvidenceType(r["link_type"]),
                    confidence=r["confidence"],
                    reason=r["reason"],
                    created_at=_iso_to_dt(r["created_at"]),
                    # M1 fields — fall back to defaults for legacy rows
                    strength_components=json.loads(_get("strength_components", "{}")),
                    strength_total=_get("strength_total"),
                    independence_group_id=_get("independence_group_id"),
                )
            )
        return results

    # ------------------ Snapshots ------------------

    def save_snapshot(self, snapshot: IncidentStateSnapshot) -> IncidentStateSnapshot:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO snapshots
                (id, incident_id, timestamp, changed_field, previous_value, new_value, reason, evidence_reference)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.id,
                    snapshot.incident_id,
                    _dt_to_iso(snapshot.timestamp),
                    snapshot.changed_field,
                    snapshot.previous_value,
                    snapshot.new_value,
                    snapshot.reason,
                    snapshot.evidence_reference,
                ),
            )
        return snapshot

    def list_snapshots_for_incident(self, incident_id: str) -> List[IncidentStateSnapshot]:
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT * FROM snapshots WHERE incident_id = ? ORDER BY timestamp ASC",
            (incident_id,),
        )
        return [
            IncidentStateSnapshot(
                id=r["id"],
                incident_id=r["incident_id"],
                timestamp=_iso_to_dt(r["timestamp"]),
                changed_field=r["changed_field"],
                previous_value=r["previous_value"],
                new_value=r["new_value"],
                reason=r["reason"],
                evidence_reference=r["evidence_reference"],
            )
            for r in cursor.fetchall()
        ]

    # ── M1: New CRUD methods ─────────────────────────────────────────────────

    # ------------------ ClaimExemplars ------------------

    def save_exemplar(self, exemplar: ClaimExemplar) -> ClaimExemplar:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO claim_exemplars
                (id, claim_type, value, text, source, version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    exemplar.id,
                    exemplar.claim_type.value,
                    exemplar.value,
                    exemplar.text,
                    exemplar.source.value,
                    exemplar.version,
                    _dt_to_iso(exemplar.created_at),
                ),
            )
        return exemplar

    def list_exemplars(
        self,
        claim_type: Optional[ClaimType] = None,
        value: Optional[str] = None,
    ) -> List[ClaimExemplar]:
        conn = self.db.get_connection()
        where_clauses = []
        params: list = []
        if claim_type:
            where_clauses.append("claim_type = ?")
            params.append(claim_type.value)
        if value:
            where_clauses.append("value = ?")
            params.append(value)
        where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        cursor = conn.execute(
            f"SELECT * FROM claim_exemplars {where} ORDER BY claim_type, value",
            params,
        )
        return [
            ClaimExemplar(
                id=r["id"],
                claim_type=ClaimType(r["claim_type"]),
                value=r["value"],
                text=r["text"],
                source=ExemplarSource(r["source"]),
                version=r["version"],
                created_at=_iso_to_dt(r["created_at"]),
            )
            for r in cursor.fetchall()
        ]

    # ------------------ UnresolvedSpans ------------------

    def save_unresolved_span(self, span: UnresolvedSpan) -> UnresolvedSpan:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO unresolved_spans
                (id, report_id, span_text, top_candidates, rejection_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    span.id,
                    span.report_id,
                    span.span_text,
                    json.dumps(span.top_candidates),
                    span.rejection_reason,
                    _dt_to_iso(span.created_at),
                ),
            )
        return span

    def list_unresolved_spans_for_report(self, report_id: str) -> List[UnresolvedSpan]:
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT * FROM unresolved_spans WHERE report_id = ? ORDER BY created_at ASC",
            (report_id,),
        )
        return [
            UnresolvedSpan(
                id=r["id"],
                report_id=r["report_id"],
                span_text=r["span_text"],
                top_candidates=json.loads(r["top_candidates"]),
                rejection_reason=r["rejection_reason"],
                created_at=_iso_to_dt(r["created_at"]),
            )
            for r in cursor.fetchall()
        ]

    # ------------------ IncidentMatches ------------------

    def save_incident_match(self, match: IncidentMatch) -> IncidentMatch:
        conn = self.db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO incident_matches
                (id, report_id, incident_id, match_reason, gate_results, semantic_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match.id,
                    match.report_id,
                    match.incident_id,
                    match.match_reason.value,
                    json.dumps(match.gate_results),
                    match.semantic_score,
                    _dt_to_iso(match.created_at),
                ),
            )
        return match

    def list_matches_for_report(self, report_id: str) -> List[IncidentMatch]:
        conn = self.db.get_connection()
        cursor = conn.execute(
            "SELECT * FROM incident_matches WHERE report_id = ? ORDER BY created_at ASC",
            (report_id,),
        )
        return [
            IncidentMatch(
                id=r["id"],
                report_id=r["report_id"],
                incident_id=r["incident_id"],
                match_reason=MatchReason(r["match_reason"]),
                gate_results=json.loads(r["gate_results"]),
                semantic_score=r["semantic_score"],
                created_at=_iso_to_dt(r["created_at"]),
            )
            for r in cursor.fetchall()
        ]
