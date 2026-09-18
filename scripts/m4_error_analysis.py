"""Detailed error analysis for isolated M4 evaluation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from crisisstate.domain.models import Report
from crisisstate.evaluation.loader import load_evaluation_dataset
from scripts.m4_evaluation import build_pipeline, execute_m4_run


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "data" / "evaluation"
EXEMPLAR_PATH = ROOT / "data" / "exemplars" / "claim_exemplars.json"
OUTPUT_PATH = ROOT / "reports" / "m4_error_analysis.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def claim_identity(claim):
    return (
        claim.claim_type.value,
        claim.subject.strip().lower(),
    )


def claim_full_identity(claim):
    return (
        claim.claim_type.value,
        claim.subject.strip().lower(),
        claim.value.strip().upper(),
    )


def analyze_report(gold_report, predicted_claims, unresolved_spans):
    gold_by_identity = {}
    pred_by_identity = {}

    for claim in gold_report.gold_claims:
        gold_by_identity.setdefault(
            (
                claim.claim_type.value,
                claim.subject.strip().lower(),
            ),
            [],
        ).append(claim)

    for claim in predicted_claims:
        pred_by_identity.setdefault(
            claim_identity(claim),
            [],
        ).append(claim)

    false_negatives = []
    false_positives = []
    value_mismatches = []

    all_identities = sorted(
        set(gold_by_identity) | set(pred_by_identity)
    )

    for identity in all_identities:
        gold_claims = gold_by_identity.get(identity, [])
        pred_claims = pred_by_identity.get(identity, [])

        matched = min(
            len(gold_claims),
            len(pred_claims),
        )

        for index in range(matched):
            gold_claim = gold_claims[index]
            pred_claim = pred_claims[index]

            if (
                gold_claim.value.strip().upper()
                != pred_claim.value.strip().upper()
            ):
                value_mismatches.append(
                    {
                        "claim_type": identity[0],
                        "subject": identity[1],
                        "gold_value": gold_claim.value,
                        "predicted_value": pred_claim.value,
                        "prediction_method": (
                            pred_claim.extraction_method.value
                        ),
                        "matched_exemplar_id": (
                            pred_claim.matched_exemplar_id
                        ),
                        "similarity_score": (
                            pred_claim.similarity_score
                        ),
                    }
                )

        for claim in gold_claims[matched:]:
            false_negatives.append(
                {
                    "claim_type": claim.claim_type.value,
                    "subject": claim.subject,
                    "value": claim.value,
                }
            )

        for claim in pred_claims[matched:]:
            false_positives.append(
                {
                    "claim_type": claim.claim_type.value,
                    "subject": claim.subject,
                    "value": claim.value,
                    "extraction_method": (
                        claim.extraction_method.value
                    ),
                    "extraction_confidence": (
                        claim.extraction_confidence
                    ),
                    "matched_exemplar_id": (
                        claim.matched_exemplar_id
                    ),
                    "similarity_score": (
                        claim.similarity_score
                    ),
                }
            )

    semantic_claims = [
        {
            "claim_type": claim.claim_type.value,
            "subject": claim.subject,
            "value": claim.value,
            "matched_exemplar_id": claim.matched_exemplar_id,
            "similarity_score": claim.similarity_score,
            "source_span": claim.source_span,
        }
        for claim in predicted_claims
        if claim.extraction_method.value == "SEMANTIC"
    ]

    lexical_claims = [
        {
            "claim_type": claim.claim_type.value,
            "subject": claim.subject,
            "value": claim.value,
        }
        for claim in predicted_claims
        if claim.extraction_method.value != "SEMANTIC"
    ]

    return {
        "gold_claim_count": len(gold_report.gold_claims),
        "predicted_claim_count": len(predicted_claims),
        "semantic_claims": semantic_claims,
        "lexical_claims": lexical_claims,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "value_mismatches": value_mismatches,
        "unresolved_spans": [
            {
                "span_text": span.span_text,
                "rejection_reason": span.rejection_reason,
                "top_candidates": span.top_candidates,
            }
            for span in unresolved_spans
        ],
    }


def main():
    dataset = load_evaluation_dataset(DATASET_DIR)
    exemplars = load_json(EXEMPLAR_PATH)

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = build_pipeline(
            exemplars,
            Path(tmp) / "m4_error_analysis.db",
        )

        predicted, _ = execute_m4_run(
            pipeline,
            dataset.reports,
        )

        # Re-run to collect per-report unresolved spans.
        unresolved_by_report = {}

        for gold_report in dataset.reports:
            report = Report(
                id=gold_report.id,
                text=gold_report.text,
                timestamp=gold_report.timestamp,
                source_type=gold_report.source_type,
                source_id=gold_report.source_id,
                latitude=gold_report.latitude,
                longitude=gold_report.longitude,
                metadata=gold_report.metadata,
            )

            result = pipeline.process_report(report)

            unresolved_by_report[report.id] = result.unresolved_spans

    per_report = {}

    total_fp = 0
    total_fn = 0
    total_value_mismatches = 0
    total_semantic = 0
    total_unresolved = 0

    for gold_report in dataset.reports:
        analysis = analyze_report(
            gold_report,
            predicted.get(gold_report.id, []),
            unresolved_by_report.get(gold_report.id, []),
        )

        per_report[gold_report.id] = analysis

        total_fp += len(analysis["false_positives"])
        total_fn += len(analysis["false_negatives"])
        total_value_mismatches += len(
            analysis["value_mismatches"]
        )
        total_semantic += len(
            analysis["semantic_claims"]
        )
        total_unresolved += len(
            analysis["unresolved_spans"]
        )

    report = {
        "evaluation": "M4 detailed error analysis",
        "report_count": len(dataset.reports),
        "totals": {
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "value_mismatches": total_value_mismatches,
            "semantic_claims": total_semantic,
            "unresolved_spans": total_unresolved,
        },
        "per_report": per_report,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("M4 Error Analysis")
    print("------------------")
    print(f"Reports:             {len(dataset.reports)}")
    print(f"False positives:     {total_fp}")
    print(f"False negatives:     {total_fn}")
    print(f"Value mismatches:    {total_value_mismatches}")
    print(f"Semantic claims:     {total_semantic}")
    print(f"Unresolved spans:    {total_unresolved}")
    print(f"Report:              {OUTPUT_PATH}")


if __name__ == "__main__":
    main()