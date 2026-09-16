"""Metrics computation for CrisisState evaluation harness."""

from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from crisisstate.domain.models import Claim, EvidenceLink, Incident, Report
from crisisstate.domain.vocabulary import EvidenceType
from crisisstate.evaluation.schemas import (
    GoldClaim,
    GoldContradiction,
    GoldModality,
    GoldReport,
    IncidentMatchingMetrics,
    MetricScore,
    PerformanceMetrics,
)


def compute_f1(precision: float, recall: float) -> float:
    """Safely compute F1 score from precision and recall."""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


def compute_claim_extraction_metrics(
    gold_reports: List[GoldReport],
    predicted_claims_by_report: Dict[str, List[Claim]],
) -> Tuple[MetricScore, float, float]:
    """
    Computes:
    1. Claim extraction Precision, Recall, F1 (matching on subject and claim_type)
    2. Claim normalization accuracy (correct value matching for true positive claims)
    3. Modality accuracy (accuracy of modality prediction against gold)
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    norm_correct = 0
    norm_total = 0

    modality_correct = 0
    modality_total = 0

    for gold_rep in gold_reports:
        rep_id = gold_rep.id
        pred_claims = predicted_claims_by_report.get(rep_id, [])
        gold_claims = gold_rep.gold_claims

        matched_pred_indices = set()
        matched_gold_indices = set()

        for g_idx, g_claim in enumerate(gold_claims):
            g_subj = g_claim.subject.strip().lower()
            g_type = g_claim.claim_type.value

            found = False
            for p_idx, p_claim in enumerate(pred_claims):
                if p_idx in matched_pred_indices:
                    continue

                p_subj = p_claim.subject.strip().lower()
                p_type = p_claim.claim_type.value

                if g_subj == p_subj and g_type == p_type:
                    matched_pred_indices.add(p_idx)
                    matched_gold_indices.add(g_idx)
                    found = True
                    total_tp += 1

                    # Check normalization (value match)
                    norm_total += 1
                    if p_claim.value.strip().upper() == g_claim.value.strip().upper():
                        norm_correct += 1

                    # Check modality (Phase 1 assumes ASSERTED by default)
                    modality_total += 1
                    # In Phase 1, all extracted claims are implicitly ASSERTED
                    predicted_modality = GoldModality.ASSERTED
                    if predicted_modality == g_claim.modality:
                        modality_correct += 1

                    break

        # Unmatched predictions are False Positives
        total_fp += len(pred_claims) - len(matched_pred_indices)
        # Unmatched gold claims are False Negatives
        total_fn += len(gold_claims) - len(matched_gold_indices)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = compute_f1(precision, recall)

    norm_acc = norm_correct / norm_total if norm_total > 0 else 0.0
    modality_acc = modality_correct / modality_total if modality_total > 0 else 0.0

    return (
        MetricScore(
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        ),
        round(norm_acc, 4),
        round(modality_acc, 4),
    )


def compute_incident_matching_metrics(
    gold_reports: List[GoldReport],
    report_to_predicted_incident: Dict[str, str],
) -> IncidentMatchingMetrics:
    """
    Computes pairwise False Merge Rate and Missed Merge Rate.
    - False Merge: Pair of reports from DIFFERENT gold incidents clustered into SAME predicted incident.
    - Missed Merge: Pair of reports from SAME gold incident clustered into DIFFERENT predicted incidents.
    """
    diff_pairs_count = 0
    false_merges = 0

    same_pairs_count = 0
    missed_merges = 0

    rep_ids = [r.id for r in gold_reports]
    gold_incident_map = {r.id: r.gold_incident_id for r in gold_reports}

    for id_a, id_b in combinations(rep_ids, 2):
        gold_a = gold_incident_map.get(id_a)
        gold_b = gold_incident_map.get(id_b)

        pred_a = report_to_predicted_incident.get(id_a)
        pred_b = report_to_predicted_incident.get(id_b)

        if gold_a != gold_b:
            diff_pairs_count += 1
            if pred_a is not None and pred_b is not None and pred_a == pred_b:
                false_merges += 1
        else:
            same_pairs_count += 1
            if pred_a is None or pred_b is None or pred_a != pred_b:
                missed_merges += 1

    false_merge_rate = (
        (false_merges / diff_pairs_count) if diff_pairs_count > 0 else 0.0
    )
    missed_merge_rate = (
        (missed_merges / same_pairs_count) if same_pairs_count > 0 else 0.0
    )

    return IncidentMatchingMetrics(
        false_merge_rate=round(false_merge_rate, 4),
        missed_merge_rate=round(missed_merge_rate, 4),
        total_distinct_pairs=diff_pairs_count,
        total_same_pairs=same_pairs_count,
    )


def compute_contradiction_metrics(
    gold_contradictions: List[GoldContradiction],
    evidence_links: List[EvidenceLink],
) -> MetricScore:
    """
    Computes Precision, Recall, and F1 for contradiction detection.
    Compares pairwise report contradiction assertions against gold pairs.
    """
    import re

    gold_pairs: Set[Tuple[str, str]] = set()
    for gc in gold_contradictions:
        if gc.relationship == "CONTRADICTS":
            gold_pairs.add(tuple(sorted([gc.report_id_a, gc.report_id_b])))

    pred_pairs: Set[Tuple[str, str]] = set()
    for link in evidence_links:
        if link.link_type == EvidenceType.CONTRADICTS:
            m = re.search(r"from Report ([\w_]+)", link.reason)
            if m:
                old_rep = m.group(1)
                pred_pairs.add(tuple(sorted([link.report_id, old_rep])))

    tp = len(pred_pairs.intersection(gold_pairs))
    fp = len(pred_pairs.difference(gold_pairs))
    fn = len(gold_pairs.difference(pred_pairs))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = compute_f1(precision, recall)

    return MetricScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )
