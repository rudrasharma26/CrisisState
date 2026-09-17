from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

EXEMPLAR_PATH = ROOT / "data" / "exemplars" / "claim_exemplars.json"
CALIBRATION_PATH = ROOT / "data" / "benchmark" / "m4_calibration_set.json"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "m4_threshold_sweep.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384

THRESHOLDS = np.arange(0.60, 0.951, 0.01)
MARGINS = np.arange(0.00, 0.201, 0.02)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    if query_norm == 0 or np.any(matrix_norms == 0):
        raise ValueError("Zero-length embedding encountered.")

    return (matrix @ query) / (matrix_norms * query_norm)


def main() -> None:
    exemplars = load_json(EXEMPLAR_PATH)
    calibration = load_json(CALIBRATION_PATH)

    if len(exemplars) != 200:
        raise AssertionError(f"Expected 200 exemplars, found {len(exemplars)}")

    if len(calibration) != 100:
        raise AssertionError(f"Expected 100 calibration probes, found {len(calibration)}")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    exemplar_texts = [
        normalize_text(item["text"])
        for item in exemplars
    ]

    calibration_texts = [
        normalize_text(item["text"])
        for item in calibration
    ]

    exemplar_embeddings = model.encode(
        exemplar_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    calibration_embeddings = model.encode(
        calibration_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if exemplar_embeddings.shape[1] != EXPECTED_DIMENSION:
        raise AssertionError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"found {exemplar_embeddings.shape[1]}"
        )

    # Precompute top-2 semantic candidates for every calibration probe.
    probe_records = []

    for probe, embedding in zip(calibration, calibration_embeddings):
        scores = cosine_scores(embedding, exemplar_embeddings)
        ranked = np.argsort(-scores)

        top1_index = int(ranked[0])
        top2_index = int(ranked[1])

        top1 = exemplars[top1_index]
        top2 = exemplars[top2_index]

        top1_score = float(scores[top1_index])
        top2_score = float(scores[top2_index])
        margin = top1_score - top2_score

        expected_pair = (
            probe["expected_claim_type"],
            probe["expected_value"],
        )

        predicted_pair = (
            top1["claim_type"],
            top1["value"],
        )

        probe_records.append(
            {
                "probe_id": probe["probe_id"],
                "expected": {
                    "claim_type": expected_pair[0],
                    "value": expected_pair[1],
                },
                "predicted_top1": {
                    "claim_type": predicted_pair[0],
                    "value": predicted_pair[1],
                    "exemplar_id": top1["id"],
                    "score": top1_score,
                },
                "top2": {
                    "claim_type": top2["claim_type"],
                    "value": top2["value"],
                    "exemplar_id": top2["id"],
                    "score": top2_score,
                },
                "margin": margin,
                "correct_top1": predicted_pair == expected_pair,
            }
        )

    sweep_results = []

    for threshold in THRESHOLDS:
        for margin_threshold in MARGINS:
            accepted = 0
            rejected = 0
            correct_accepts = 0
            wrong_accepts = 0

            for record in probe_records:
                score = record["predicted_top1"]["score"]
                margin = record["margin"]

                is_accepted = (
                    score >= threshold
                    and margin >= margin_threshold
                )

                if is_accepted:
                    accepted += 1

                    if record["correct_top1"]:
                        correct_accepts += 1
                    else:
                        wrong_accepts += 1
                else:
                    rejected += 1

            total = len(probe_records)

            accepted_accuracy = (
                correct_accepts / accepted
                if accepted > 0
                else 0.0
            )

            accepted_rate = accepted / total

            wrong_acceptance_rate = (
                wrong_accepts / total
            )

            correct_coverage = (
                correct_accepts / total
            )

            sweep_results.append(
                {
                    "threshold": round(float(threshold), 2),
                    "margin_threshold": round(float(margin_threshold), 2),
                    "accepted": accepted,
                    "rejected": rejected,
                    "accepted_rate": accepted_rate,
                    "correct_accepts": correct_accepts,
                    "wrong_accepts": wrong_accepts,
                    "accepted_accuracy": accepted_accuracy,
                    "wrong_acceptance_rate": wrong_acceptance_rate,
                    "correct_coverage": correct_coverage,
                }
            )

    report = {
        "model": MODEL_NAME,
        "embedding_dimension": EXPECTED_DIMENSION,
        "device": "cpu",
        "exemplar_count": len(exemplars),
        "calibration_probe_count": len(calibration),
        "similarity": "cosine",
        "threshold_range": {
            "start": float(THRESHOLDS[0]),
            "end": float(THRESHOLDS[-1]),
            "step": 0.01,
        },
        "margin_range": {
            "start": float(MARGINS[0]),
            "end": float(MARGINS[-1]),
            "step": 0.02,
        },
        "probe_results": probe_records,
        "sweep": sweep_results,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("M4 Threshold Sweep")
    print("------------------")
    print(f"Model:       {MODEL_NAME}")
    print(f"Exemplars:   {len(exemplars)}")
    print(f"Calibration: {len(calibration)}")
    print(f"Thresholds:  {len(THRESHOLDS)}")
    print(f"Margins:     {len(MARGINS)}")
    print(f"Configurations tested: {len(sweep_results)}")
    print(f"Report:      {REPORT_PATH}")


if __name__ == "__main__":
    main()