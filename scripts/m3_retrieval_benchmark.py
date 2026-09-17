from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]

EXEMPLAR_PATH = ROOT / "data" / "exemplars" / "claim_exemplars.json"
PROBE_PATH = ROOT / "data" / "benchmark" / "m3_retrieval_probes.json"
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "m3_retrieval_report.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384
TOP_K_VALUES = (1, 3, 5)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


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
    probes = load_json(PROBE_PATH)

    if len(exemplars) != 200:
        raise AssertionError(f"Expected 200 exemplars, found {len(exemplars)}")

    if len(probes) != 60:
        raise AssertionError(f"Expected 60 probes, found {len(probes)}")

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    exemplar_texts = [normalize_text(x["text"]) for x in exemplars]
    probe_texts = [normalize_text(x["text"]) for x in probes]

    exemplar_embeddings = model.encode(
        exemplar_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    probe_embeddings = model.encode(
        probe_texts,
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

    rankings = []
    top1_confusion = Counter()

    recall_hits = {k: 0 for k in TOP_K_VALUES}
    reciprocal_ranks = []

    for probe, probe_embedding in zip(probes, probe_embeddings):
        scores = cosine_scores(probe_embedding, exemplar_embeddings)
        ranked_indices = np.argsort(-scores)

        expected_pair = (
            probe["expected_claim_type"],
            probe["expected_value"],
        )

        first_correct_rank = None
        top_results = []

        for rank_index, exemplar_index in enumerate(ranked_indices[:5], start=1):
            exemplar = exemplars[int(exemplar_index)]

            pair = (
                exemplar["claim_type"],
                exemplar["value"],
            )

            top_results.append(
                {
                    "rank": rank_index,
                    "exemplar_id": exemplar["id"],
                    "claim_type": exemplar["claim_type"],
                    "value": exemplar["value"],
                    "score": float(scores[exemplar_index]),
                }
            )

            if first_correct_rank is None and pair == expected_pair:
                first_correct_rank = rank_index

        for k in TOP_K_VALUES:
            if first_correct_rank is not None and first_correct_rank <= k:
                recall_hits[k] += 1

        if first_correct_rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / first_correct_rank)

        top1 = top_results[0]

        predicted_pair = (
            top1["claim_type"],
            top1["value"],
        )

        if predicted_pair != expected_pair:
            top1_confusion[
                (
                    f"{expected_pair[0]}:{expected_pair[1]}",
                    f"{predicted_pair[0]}:{predicted_pair[1]}",
                )
            ] += 1

        rankings.append(
            {
                "probe_id": probe["probe_id"],
                "expected_claim_type": expected_pair[0],
                "expected_value": expected_pair[1],
                "top_results": top_results,
                "first_correct_rank": first_correct_rank,
            }
        )

    total = len(probes)

    recall = {
        f"recall_at_{k}": recall_hits[k] / total
        for k in TOP_K_VALUES
    }

    mrr = float(np.mean(reciprocal_ranks))

    wrong_top1_rate = sum(
        1
        for result in rankings
        if result["first_correct_rank"] != 1
    ) / total

    report = {
        "model": MODEL_NAME,
        "embedding_dimension": int(exemplar_embeddings.shape[1]),
        "device": "cpu",
        "exemplar_count": len(exemplars),
        "probe_count": len(probes),
        "normalization": "lowercase + whitespace normalization",
        "similarity": "cosine",
        **recall,
        "mrr": mrr,
        "wrong_value_retrieval_rate": wrong_top1_rate,
        "top1_confusions": [
            {
                "expected": expected,
                "predicted": predicted,
                "count": count,
            }
            for (expected, predicted), count in top1_confusion.most_common()
        ],
        "rankings": rankings,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("M3 Retrieval Benchmark")
    print("----------------------")
    print(f"Model:              {MODEL_NAME}")
    print(f"Embedding dimension:{exemplar_embeddings.shape[1]}")
    print(f"Exemplars:          {len(exemplars)}")
    print(f"Probes:             {len(probes)}")
    print(f"Recall@1:           {recall['recall_at_1']:.4f}")
    print(f"Recall@3:           {recall['recall_at_3']:.4f}")
    print(f"Recall@5:           {recall['recall_at_5']:.4f}")
    print(f"MRR:                {mrr:.4f}")
    print(f"Wrong top-1 rate:   {wrong_top1_rate:.4f}")
    print(f"Report:             {REPORT_PATH}")


if __name__ == "__main__":
    main()