"""Isolated M4 semantic processing pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from crisisstate.domain.models import Claim, Report, UnresolvedSpan
from crisisstate.domain.vocabulary import ClaimStatus, ExtractionMethod
from crisisstate.engine.extractor import extract_claims, extract_entity
from crisisstate.semantic.adjudicator import SemanticAdjudicator
from crisisstate.semantic.candidate_generator import SemanticCandidateGenerator


@dataclass
class M4PipelineResult:
    claims: List[Claim] = field(default_factory=list)
    unresolved_spans: List[UnresolvedSpan] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


class M4SemanticPipeline:
    """Run semantic processing without modifying the production pipeline."""

    _SPAN_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")

    def __init__(
        self,
        candidate_generator: SemanticCandidateGenerator,
        adjudicator: SemanticAdjudicator,
    ) -> None:
        self.candidate_generator = candidate_generator
        self.adjudicator = adjudicator

    def process_report(self, report: Report) -> M4PipelineResult:
        result = M4PipelineResult()

        for span_text, start, end in self._segment(report.text):
            span_report = Report(
                id=report.id,
                text=span_text,
                timestamp=report.timestamp,
                source_type=report.source_type,
                source_id=report.source_id,
                latitude=report.latitude,
                longitude=report.longitude,
                metadata=report.metadata,
            )

            lexical_claims = extract_claims(span_report)

            source_span = {
                "start": start,
                "end": end,
                "text": span_text,
            }

            if lexical_claims:
                for claim in lexical_claims:
                    claim.source_span = source_span

                    result.claims.append(claim)

                    result.audit_trail.append(
                        {
                            "source_span": source_span,
                            "decision": "LEXICAL_PRESERVED",
                            "claim_type": claim.claim_type.value,
                            "value": claim.value,
                            "rule_applied": "PRESERVE_PHASE1_LEXICAL_CLAIM",
                        }
                    )

                continue

            semantic_result = self.candidate_generator.generate(
                span_text,
                top_k=5,
            )

            adjudication = self.adjudicator.adjudicate(
                span_text,
                semantic_result,
            )

            result.audit_trail.append(
                {
                    "source_span": source_span,
                    "semantic_candidates": semantic_result.get(
                        "candidates",
                        [],
                    ),
                    "adjudication": adjudication,
                }
            )

            if adjudication["status"] == "ACCEPTED":
                claim = Claim(
                    report_id=report.id,
                    claim_type=adjudication["claim_type"],
                    subject=extract_entity(span_report),
                    value=adjudication["value"],
                    timestamp=report.timestamp,
                    status=ClaimStatus.ACTIVE,
                    confidence=adjudication["similarity_score"],
                    supporting_evidence=[report.id],
                    contradicting_evidence=[],
                    extraction_method=ExtractionMethod.SEMANTIC,
                    extraction_confidence=adjudication["similarity_score"],
                    matched_exemplar_id=adjudication["matched_exemplar_id"],
                    similarity_score=adjudication["similarity_score"],
                    source_span=source_span,
                    candidate_values=[
                        f"{candidate['claim_type']}:{candidate['value']}"
                        for candidate in semantic_result.get(
                            "candidates",
                            [],
                        )
                    ],
                )

                result.claims.append(claim)

            else:
                candidates = semantic_result.get("candidates", [])

                unresolved = UnresolvedSpan(
                    report_id=report.id,
                    span_text=span_text,
                    top_candidates=[
                        {
                            "claim_type": candidate["claim_type"],
                            "value": candidate["value"],
                            "score": candidate["similarity_score"],
                        }
                        for candidate in candidates
                    ],
                    rejection_reason=adjudication.get(
                        "rule_applied",
                        "UNRESOLVED",
                    ),
                )

                result.unresolved_spans.append(unresolved)

        return result

    @classmethod
    def _segment(cls, text: str) -> List[Tuple[str, int, int]]:
        spans: List[Tuple[str, int, int]] = []

        for match in cls._SPAN_RE.finditer(text):
            raw = match.group(0)
            stripped = raw.strip()

            if not stripped:
                continue

            leading = len(raw) - len(raw.lstrip())
            start = match.start() + leading
            end = start + len(stripped)

            spans.append((stripped, start, end))

        return spans