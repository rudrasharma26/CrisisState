"""Evaluation runner and CLI entrypoint for CrisisState evaluation harness."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from crisisstate.domain.models import Claim, EvidenceLink, Incident, Report
from crisisstate.engine.pipeline import IncidentPipeline, PipelineResult
from crisisstate.evaluation.baseline import get_frozen_phase1_baseline
from crisisstate.evaluation.loader import EvaluationDataset, load_evaluation_dataset
from crisisstate.evaluation.metrics import (
    compute_claim_extraction_metrics,
    compute_contradiction_metrics,
    compute_incident_matching_metrics,
)
from crisisstate.evaluation.reports import format_terminal_report, save_json_report
from crisisstate.evaluation.schemas import (
    EvaluationSummary,
    GoldReport,
    PerformanceMetrics,
)
from crisisstate.storage.database import Database
from crisisstate.storage.repository import CrisisRepository


class EvaluationRunner:
    """Executes evaluation runs against a dataset and computes metrics."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.dataset = load_evaluation_dataset(self.dataset_dir)

    def _execute_single_run(
        self,
    ) -> Tuple[
        Dict[str, List[Claim]],
        Dict[str, str],
        List[EvidenceLink],
        List[Incident],
        float,
    ]:
        """Runs the entire dataset through a fresh in-memory pipeline."""
        db = Database(":memory:")
        repo = CrisisRepository(db)
        pipeline = IncidentPipeline(repo)

        predicted_claims_by_report: Dict[str, List[Claim]] = {}
        report_to_incident: Dict[str, str] = {}
        all_evidence_links: List[EvidenceLink] = []

        start_time = time.perf_counter()

        for gold_rep in self.dataset.reports:
            report_obj = Report(
                id=gold_rep.id,
                text=gold_rep.text,
                timestamp=gold_rep.timestamp,
                source_type=gold_rep.source_type,
                source_id=gold_rep.source_id,
                latitude=gold_rep.latitude,
                longitude=gold_rep.longitude,
                metadata=gold_rep.metadata,
            )

            result: PipelineResult = pipeline.process_report(report_obj)

            predicted_claims_by_report[gold_rep.id] = result.extracted_claims
            report_to_incident[gold_rep.id] = result.incident.id
            all_evidence_links.extend(result.evidence_links)

        total_runtime = time.perf_counter() - start_time
        all_incidents = repo.list_incidents()

        return (
            predicted_claims_by_report,
            report_to_incident,
            all_evidence_links,
            all_incidents,
            total_runtime,
        )

    @staticmethod
    def _canonical_partition(rep_to_inc: Dict[str, str]) -> Dict[str, str]:
        """Maps each report to the lexicographically first report in its cluster."""
        inc_to_reps: Dict[str, List[str]] = {}
        for r_id, inc_id in rep_to_inc.items():
            inc_to_reps.setdefault(inc_id, []).append(r_id)
        canonical_map: Dict[str, str] = {}
        for reps in inc_to_reps.values():
            leader = sorted(reps)[0]
            for r in reps:
                canonical_map[r] = leader
        return canonical_map

    def evaluate(self, verify_determinism: bool = True) -> EvaluationSummary:
        """Run evaluation, check determinism, and compute all metrics."""
        (
            pred_claims_1,
            rep_inc_1,
            ev_links_1,
            incidents_1,
            runtime_1,
        ) = self._execute_single_run()

        # Determinism check
        determinism_status = "PASS"
        if verify_determinism:
            (
                pred_claims_2,
                rep_inc_2,
                ev_links_2,
                incidents_2,
                _,
            ) = self._execute_single_run()

            # Verify identical incident partitions
            p1 = self._canonical_partition(rep_inc_1)
            p2 = self._canonical_partition(rep_inc_2)
            if p1 != p2:
                determinism_status = "FAIL"
            elif len(ev_links_1) != len(ev_links_2):
                determinism_status = "FAIL"
            else:
                # Compare extracted claims for each report
                for r_id in self.dataset.reports_by_id:
                    c1 = [
                        f"{c.claim_type.value}:{c.subject}:{c.value}:{c.status.value}"
                        for c in pred_claims_1.get(r_id, [])
                    ]
                    c2 = [
                        f"{c.claim_type.value}:{c.subject}:{c.value}:{c.status.value}"
                        for c in pred_claims_2.get(r_id, [])
                    ]
                    if c1 != c2:
                        determinism_status = "FAIL"
                        break

        # Compute metrics
        claim_score, norm_acc, modality_acc = compute_claim_extraction_metrics(
            self.dataset.reports, pred_claims_1
        )

        incident_metrics = compute_incident_matching_metrics(
            self.dataset.reports, rep_inc_1
        )

        contradiction_score = compute_contradiction_metrics(
            self.dataset.contradictions, ev_links_1
        )

        total_reports = len(self.dataset.reports)
        avg_latency = (
            (runtime_1 / total_reports) * 1000.0 if total_reports > 0 else 0.0
        )

        # Count categories
        cat_counts: Dict[str, int] = {}
        for r in self.dataset.reports:
            cat_counts[r.category.value] = cat_counts.get(r.category.value, 0) + 1

        baseline_meta = get_frozen_phase1_baseline()

        return EvaluationSummary(
            system_version=baseline_meta["baseline_tag"],
            dataset_version="v1.0-seed-corpus",
            total_reports=total_reports,
            category_counts=cat_counts,
            claim_extraction=claim_score,
            claim_normalization_accuracy=norm_acc,
            modality_accuracy=modality_acc,
            incident_matching=incident_metrics,
            contradiction_detection=contradiction_score,
            determinism=determinism_status,
            performance=PerformanceMetrics(
                total_runtime_seconds=round(runtime_1, 4),
                avg_latency_per_report_ms=round(avg_latency, 2),
            ),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run CrisisState Evaluation Harness against frozen baseline"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="data/evaluation",
        help="Path to evaluation dataset directory (default: data/evaluation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="baseline_report.json",
        help="Path to output JSON evaluation report (default: baseline_report.json)",
    )
    parser.add_argument(
        "--save-baseline-config",
        type=str,
        default=None,
        help="Optional path to export machine-readable frozen baseline configuration JSON",
    )
    parser.add_argument(
        "--skip-determinism",
        action="store_true",
        help="Skip replay determinism verification",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{args.dataset_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running CrisisState Evaluation Harness on '{dataset_path}'...")
    runner = EvaluationRunner(dataset_path)
    summary = runner.evaluate(verify_determinism=not args.skip_determinism)

    # Print terminal output
    print(format_terminal_report(summary))

    # Save output JSON
    baseline_info = get_frozen_phase1_baseline()
    save_json_report(summary, Path(args.output), extra_metadata=baseline_info)
    print(f"\nMachine-readable evaluation report saved to: {args.output}")

    if args.save_baseline_config:
        config_path = Path(args.save_baseline_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(baseline_info, f, indent=2)
        print(f"Baseline configuration saved to: {args.save_baseline_config}")


if __name__ == "__main__":
    main()
