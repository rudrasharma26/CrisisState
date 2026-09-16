"""Reporting and formatting module for evaluation results."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from crisisstate.evaluation.schemas import EvaluationSummary


def format_terminal_report(summary: EvaluationSummary) -> str:
    """Format evaluation summary into the standard human-readable terminal view."""
    lines = [
        "=" * 60,
        "# CrisisState Evaluation Report",
        "=" * 60,
        f"Baseline: {summary.system_version}",
        f"Dataset:  {summary.dataset_version}",
        f"Generated At: {summary.generated_at.isoformat()}",
        "",
        "Dataset Summary:",
        f"  Total Reports: {summary.total_reports}",
        "  Category Breakdown:",
    ]

    for cat, count in sorted(summary.category_counts.items()):
        lines.append(f"    - {cat:<26}: {count}")

    lines.extend(
        [
            "",
            "Claim Extraction:",
            f"  Precision: {summary.claim_extraction.precision:.4f}",
            f"  Recall:    {summary.claim_extraction.recall:.4f}",
            f"  F1:        {summary.claim_extraction.f1:.4f}",
            "",
            "Claim Normalization:",
            f"  Accuracy:  {summary.claim_normalization_accuracy * 100:.1f}%",
            "",
            "Modality Classification (Phase 1 Baseline):",
            f"  Accuracy:  {summary.modality_accuracy * 100:.1f}%",
            "",
            "Incident Matching:",
            f"  False Merge Rate:  {summary.incident_matching.false_merge_rate * 100:.2f}% ({summary.incident_matching.total_distinct_pairs} distinct pairs)",
            f"  Missed Merge Rate: {summary.incident_matching.missed_merge_rate * 100:.2f}% ({summary.incident_matching.total_same_pairs} same pairs)",
            "",
            "Contradiction Detection:",
            f"  Precision: {summary.contradiction_detection.precision:.4f}",
            f"  Recall:    {summary.contradiction_detection.recall:.4f}",
            f"  F1:        {summary.contradiction_detection.f1:.4f}",
            "",
            "Replay Determinism:",
            f"  Status:    {summary.determinism}",
            "",
            "Performance:",
            f"  Total Runtime:       {summary.performance.total_runtime_seconds:.3f} s",
            f"  Latency Per Report:  {summary.performance.avg_latency_per_report_ms:.2f} ms",
            "=" * 60,
        ]
    )
    return "\n".join(lines)


def save_json_report(
    summary: EvaluationSummary,
    output_path: Path,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save the evaluation summary to a machine-readable JSON file."""
    data = summary.model_dump(mode="json")
    if extra_metadata:
        data["metadata"] = extra_metadata

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
