import json
from pathlib import Path


REPORT_PATH = Path("reports/m4_error_analysis.json")


def main():
    with REPORT_PATH.open("r", encoding="utf-8") as f:
        report = json.load(f)

    print("\n=== FALSE POSITIVES ===")
    for report_id, data in report["per_report"].items():
        if data["false_positives"]:
            print(f"\n[{report_id}]")
            for item in data["false_positives"]:
                print(
                    f"  {item['claim_type']}:{item['value']} | "
                    f"method={item['extraction_method']} | "
                    f"score={item['similarity_score']}"
                )

    print("\n=== FALSE NEGATIVES ===")
    for report_id, data in report["per_report"].items():
        if data["false_negatives"]:
            print(f"\n[{report_id}]")
            for item in data["false_negatives"]:
                print(
                    f"  {item['claim_type']}:{item['value']}"
                )

    print("\n=== VALUE MISMATCHES ===")
    for report_id, data in report["per_report"].items():
        if data["value_mismatches"]:
            print(f"\n[{report_id}]")
            for item in data["value_mismatches"]:
                print(
                    f"  {item['claim_type']} | "
                    f"GOLD={item['gold_value']} | "
                    f"PRED={item['predicted_value']} | "
                    f"method={item['prediction_method']} | "
                    f"score={item['similarity_score']}"
                )

    print("\n=== SEMANTIC CLAIMS ===")
    for report_id, data in report["per_report"].items():
        if data["semantic_claims"]:
            print(f"\n[{report_id}]")
            for item in data["semantic_claims"]:
                print(
                    f"  {item['claim_type']}:{item['value']} | "
                    f"exemplar={item['matched_exemplar_id']} | "
                    f"score={item['similarity_score']}"
                )

    print("\n=== UNRESOLVED SPANS ===")
    for report_id, data in report["per_report"].items():
        if data["unresolved_spans"]:
            print(f"\n[{report_id}]")
            for item in data["unresolved_spans"]:
                print(
                    f'  "{item["span_text"]}" | '
                    f'reason={item["rejection_reason"]}'
                )

                for candidate in item["top_candidates"]:
                    print(
                        f"    -> "
                        f"{candidate['claim_type']}:"
                        f"{candidate['value']} "
                        f"({candidate['score']})"
                    )


if __name__ == "__main__":
    main()