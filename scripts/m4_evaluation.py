"""Evaluate isolated M4 semantic extraction against the frozen M0 corpus."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from crisisstate.domain.models import Report
from crisisstate.evaluation.loader import load_evaluation_dataset
from crisisstate.evaluation.metrics import compute_claim_extraction_metrics
from crisisstate.semantic.adjudicator import SemanticAdjudicator
from crisisstate.semantic.candidate_generator import SemanticCandidateGenerator
from crisisstate.semantic.embedding_cache import EmbeddingCache
from crisisstate.semantic.embedding_service import EmbeddingService
from crisisstate.semantic.exemplar_search import ExemplarSearch
from crisisstate.semantic.m4_pipeline import M4SemanticPipeline
from crisisstate.semantic.search_adapter import ClaimExemplarSearchAdapter



ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "data" / "evaluation"
EXEMPLAR_PATH = ROOT / "data" / "exemplars" / "claim_exemplars.json"
BASELINE_CONFIG_PATH = ROOT / "baseline_config.json"

REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "m4_evaluation_report.json"

THRESHOLD = 0.67
MARGIN = 0.04


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_pipeline(
    exemplars: List[Dict[str, Any]],
    cache_location: Path,
) -> M4SemanticPipeline:
    """Build an isolated M4 pipeline with a file-backed temporary cache."""

    service = EmbeddingService()

    cache = EmbeddingCache(cache_location)

    search_engine = ExemplarSearch(
        exemplars,
        service,
        cache,
    )

    search_adapter = ClaimExemplarSearchAdapter(
        search_engine,
        exemplars,
    )

    generator = SemanticCandidateGenerator(
        search_adapter,
        similarity_threshold=THRESHOLD,
        margin_threshold=MARGIN,
    )

    adjudicator = SemanticAdjudicator(
        similarity_threshold=THRESHOLD,
        margin_threshold=MARGIN,
    )

    return M4SemanticPipeline(
        generator,
        adjudicator,
    )

def execute_m4_run(
    pipeline: M4SemanticPipeline,
    reports,
):
    predicted_claims_by_report = {}
    unresolved_count = 0
    semantic_claim_count = 0
    lexical_claim_count = 0

    for gold_report in reports:
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

        predicted_claims_by_report[gold_report.id] = result.claims

        unresolved_count += len(result.unresolved_spans)

        for claim in result.claims:
            if claim.extraction_method.value == "SEMANTIC":
                semantic_claim_count += 1
            else:
                lexical_claim_count += 1

    return (
        predicted_claims_by_report,
        {
            "unresolved_spans": unresolved_count,
            "semantic_claims": semantic_claim_count,
            "lexical_claims": lexical_claim_count,
        },
    )


def canonical_claims(predicted):
    result = {}

    for report_id, claims in predicted.items():
        result[report_id] = [
            (
                claim.claim_type.value,
                claim.subject,
                claim.value,
                claim.extraction_method.value,
            )
            for claim in claims
        ]

    return result


def main() -> None:
    dataset = load_evaluation_dataset(DATASET_DIR)
    exemplars = load_json(EXEMPLAR_PATH)

    if len(exemplars) != 200:
        raise AssertionError(
            f"Expected 200 exemplars, found {len(exemplars)}"
        )

    if not BASELINE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Frozen baseline configuration not found: {BASELINE_CONFIG_PATH}"
        )

    baseline_config = load_json(BASELINE_CONFIG_PATH)

    if baseline_config.get("baseline_tag") != "v1.0.0-phase1-frozen":
        raise AssertionError(
            "baseline_config.json is not tagged as the frozen Phase 1 baseline"
        )

    # Locked Phase 1 baseline metrics from the M0 freeze.
    baseline = {
        "claim_extraction": {
            "precision": 0.8375,
            "recall": 0.6634,
            "f1": 0.7403,
        },
        "claim_normalization_accuracy": 0.9100,
        "modality_accuracy": 0.8060,
    }

    with tempfile.TemporaryDirectory() as tmp:
        first_cache = Path(tmp) / "m4_embeddings_first.db"
        second_cache = Path(tmp) / "m4_embeddings_second.db"

        pipeline = build_pipeline(
            exemplars,
            first_cache,
        )

        predicted_1, stats = execute_m4_run(
            pipeline,
            dataset.reports,
        )

        pipeline_2 = build_pipeline(
            exemplars,
            second_cache,
        )

        predicted_2, _ = execute_m4_run(
            pipeline_2,
            dataset.reports,
        )

    claim_score, normalization_accuracy, modality_accuracy = (
        compute_claim_extraction_metrics(
            dataset.reports,
            predicted_1,
        )
    )

    deterministic = (
        canonical_claims(predicted_1)
        == canonical_claims(predicted_2)
    )

    baseline_claim = baseline["claim_extraction"]
    baseline_norm = baseline["claim_normalization_accuracy"]
    baseline_modality = baseline["modality_accuracy"]

    report = {
        "evaluation": "M4 isolated semantic extraction",
        "baseline_tag": "v1.0.0-phase1-frozen",
        "dataset_version": "v1.0-seed-corpus",
        "threshold": THRESHOLD,
        "margin_threshold": MARGIN,
        "exemplar_count": len(exemplars),
        "report_count": len(dataset.reports),
        "m4_claim_extraction": {
            "precision": claim_score.precision,
            "recall": claim_score.recall,
            "f1": claim_score.f1,
        },
        "m4_claim_normalization_accuracy": normalization_accuracy,
        "m4_modality_accuracy": modality_accuracy,
        "baseline_claim_extraction": baseline_claim,
        "baseline_claim_normalization_accuracy": baseline_norm,
        "baseline_modality_accuracy": baseline_modality,
        "delta": {
            "precision": round(
                claim_score.precision - baseline_claim["precision"],
                4,
            ),
            "recall": round(
                claim_score.recall - baseline_claim["recall"],
                4,
            ),
            "f1": round(
                claim_score.f1 - baseline_claim["f1"],
                4,
            ),
            "normalization_accuracy": round(
                normalization_accuracy - baseline_norm,
                4,
            ),
            "modality_accuracy": round(
                modality_accuracy - baseline_modality,
                4,
            ),
        },
        "semantic_claims": stats["semantic_claims"],
        "lexical_claims": stats["lexical_claims"],
        "unresolved_spans": stats["unresolved_spans"],
        "determinism": "PASS" if deterministic else "FAIL",
        "incident_matching": (
            "NOT_REEVALUATED — M4 remains isolated; "
            "production incident matcher is unchanged"
        ),
        "contradiction_detection": (
            "NOT_REEVALUATED — M4 remains isolated; "
            "production contradiction engine is unchanged"
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("M4 Isolated Evaluation")
    print("----------------------")
    print(f"Reports:              {len(dataset.reports)}")
    print(f"Exemplars:            {len(exemplars)}")
    print(f"Threshold:            {THRESHOLD}")
    print(f"Margin:               {MARGIN}")
    print()
    print("M4 Claim Extraction")
    print(f"Precision:             {claim_score.precision:.4f}")
    print(f"Recall:                {claim_score.recall:.4f}")
    print(f"F1:                    {claim_score.f1:.4f}")
    print(f"Normalization:         {normalization_accuracy:.4f}")
    print(f"Modality:              {modality_accuracy:.4f}")
    print()
    print("Delta vs frozen Phase 1")
    print(f"Precision:             {report['delta']['precision']:+.4f}")
    print(f"Recall:                {report['delta']['recall']:+.4f}")
    print(f"F1:                    {report['delta']['f1']:+.4f}")
    print(
        "Normalization:         "
        f"{report['delta']['normalization_accuracy']:+.4f}"
    )
    print(
        "Modality:              "
        f"{report['delta']['modality_accuracy']:+.4f}"
    )
    print()
    print(f"Semantic claims:       {stats['semantic_claims']}")
    print(f"Lexical claims:        {stats['lexical_claims']}")
    print(f"Unresolved spans:      {stats['unresolved_spans']}")
    print(f"Determinism:            {report['determinism']}")
    print(f"Report:                {REPORT_PATH}")


if __name__ == "__main__":
    main()