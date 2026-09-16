"""Tests for the Phase 2 M0 Evaluation Harness and Frozen Baseline."""

from pathlib import Path
import pytest

from crisisstate.evaluation.baseline import get_frozen_phase1_baseline
from crisisstate.evaluation.loader import EvaluationDataset, load_evaluation_dataset
from crisisstate.evaluation.metrics import (
    compute_claim_extraction_metrics,
    compute_contradiction_metrics,
    compute_f1,
    compute_incident_matching_metrics,
)
from crisisstate.evaluation.reports import format_terminal_report, save_json_report
from crisisstate.evaluation.runner import EvaluationRunner
from crisisstate.evaluation.schemas import (
    GoldClaim,
    GoldContradiction,
    GoldIncident,
    GoldModality,
    GoldReport,
    TestCategory,
)


@pytest.fixture
def dataset_dir() -> Path:
    return Path("data/evaluation")


def test_frozen_baseline_config():
    """Verify machine-readable frozen baseline configuration structure."""
    baseline = get_frozen_phase1_baseline()
    assert baseline["baseline_tag"] == "v1.0.0-phase1-frozen"
    assert baseline["phase"] == 1
    assert "environment" in baseline
    assert "dependencies" in baseline
    assert "claim_types" in baseline
    assert len(baseline["claim_types"]) == 7
    assert "ROAD_ACCESS" in baseline["claim_types"]
    assert "contradiction_pairs" in baseline
    assert "engine_configuration" in baseline


def test_evaluation_dataset_loading_and_integrity(dataset_dir: Path):
    """Verify that the evaluation dataset loads without errors and satisfies integrity checks."""
    assert dataset_dir.exists()
    dataset = load_evaluation_dataset(dataset_dir)

    assert len(dataset.incidents) >= 8
    assert len(dataset.reports) >= 75
    assert len(dataset.claims) >= 75
    assert len(dataset.contradictions) >= 5

    # Check categories coverage
    categories = {r.category for r in dataset.reports}
    assert TestCategory.EXACT_LEXICAL in categories
    assert TestCategory.PARAPHRASE in categories
    assert TestCategory.NEGATION in categories
    assert TestCategory.HEDGING in categories
    assert TestCategory.QUALIFIED_SCOPED in categories
    assert TestCategory.GEO_CLOSE_DISTINCT in categories
    assert TestCategory.SEMANTIC_SIMILAR_DISTINCT in categories
    assert TestCategory.CONTRADICTION in categories
    assert TestCategory.TEMPORAL_SUPERSESSION in categories

    # Verify no integrity validation errors
    errors = dataset.validate_integrity()
    assert len(errors) == 0


def test_metrics_computation_unit():
    """Verify metric computation formulas and edge cases."""
    # F1 edge cases
    assert compute_f1(0.0, 0.0) == 0.0
    assert compute_f1(1.0, 1.0) == 1.0
    assert round(compute_f1(0.8, 0.8), 2) == 0.8

    # Incident matching metrics
    gold_reps = [
        GoldReport(
            id="r1",
            text="t1",
            timestamp="2026-09-15T08:00:00Z",
            location="loc1",
            latitude=28.0,
            longitude=77.0,
            gold_incident_id="inc_A",
            category=TestCategory.EXACT_LEXICAL,
        ),
        GoldReport(
            id="r2",
            text="t2",
            timestamp="2026-09-15T08:05:00Z",
            location="loc1",
            latitude=28.0,
            longitude=77.0,
            gold_incident_id="inc_A",
            category=TestCategory.EXACT_LEXICAL,
        ),
        GoldReport(
            id="r3",
            text="t3",
            timestamp="2026-09-15T08:10:00Z",
            location="loc2",
            latitude=28.5,
            longitude=77.5,
            gold_incident_id="inc_B",
            category=TestCategory.EXACT_LEXICAL,
        ),
    ]

    # Case 1: Perfect clustering
    pred_map_perfect = {"r1": "cluster_1", "r2": "cluster_1", "r3": "cluster_2"}
    m_perfect = compute_incident_matching_metrics(gold_reps, pred_map_perfect)
    assert m_perfect.false_merge_rate == 0.0
    assert m_perfect.missed_merge_rate == 0.0

    # Case 2: False merge (r1, r2, r3 all merged into cluster_1)
    pred_map_false_merge = {"r1": "cluster_1", "r2": "cluster_1", "r3": "cluster_1"}
    m_fm = compute_incident_matching_metrics(gold_reps, pred_map_false_merge)
    assert m_fm.false_merge_rate > 0.0
    assert m_fm.missed_merge_rate == 0.0

    # Case 3: Missed merge (r1 and r2 split)
    pred_map_missed = {"r1": "cluster_1", "r2": "cluster_2", "r3": "cluster_3"}
    m_mm = compute_incident_matching_metrics(gold_reps, pred_map_missed)
    assert m_mm.false_merge_rate == 0.0
    assert m_mm.missed_merge_rate > 0.0


def test_evaluation_runner_execution(dataset_dir: Path, tmp_path: Path):
    """Verify end-to-end evaluation run, determinism, and report outputs."""
    runner = EvaluationRunner(dataset_dir)
    summary = runner.evaluate(verify_determinism=True)

    assert summary.total_reports >= 75
    assert summary.determinism == "PASS"
    assert summary.claim_extraction.precision > 0.0
    assert summary.claim_extraction.recall > 0.0
    assert summary.performance.total_runtime_seconds > 0.0
    assert summary.performance.avg_latency_per_report_ms > 0.0

    # Verify terminal report formatting
    term_report = format_terminal_report(summary)
    assert "# CrisisState Evaluation Report" in term_report
    assert "Determinism:" in term_report
    assert "PASS" in term_report

    # Verify JSON output saving
    out_json = tmp_path / "eval_out.json"
    save_json_report(summary, out_json)
    assert out_json.exists()
