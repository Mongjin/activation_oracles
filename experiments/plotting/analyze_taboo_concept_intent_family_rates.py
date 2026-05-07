import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


FAMILY_PATTERNS = {
    "concealment": [
        r"\bhide\b",
        r"\bhidden\b",
        r"\bconceal\b",
        r"\bconcealed\b",
        r"\bconcealment\b",
        r"\bmask\b",
        r"\bmasked\b",
        r"\bmasking\b",
        r"\bobscure\b",
        r"\bobscured\b",
        r"\bobfuscate\b",
        r"\bobfuscation\b",
        r"\bsecretive\b",
        r"\bkeeping .* hidden\b",
    ],
    "suppression": [
        r"\bsuppress\b",
        r"\bsuppressed\b",
        r"\bsuppression\b",
        r"\bfilter\b",
        r"\bfiltered\b",
        r"\bcensor\b",
        r"\bcensored\b",
        r"\bblock\b",
        r"\bblocked\b",
        r"\bprevent\b",
        r"\bprevented\b",
        r"\binhibit\b",
        r"\binhibited\b",
        r"\bconstraint\b",
    ],
    "non_disclosure": [
        r"\bwithhold\b",
        r"\bwithholding\b",
        r"\bdoes not reveal\b",
        r"\bdo not reveal\b",
        r"\bnot reveal\b",
        r"\bcannot reveal\b",
        r"\bcan't reveal\b",
        r"\brefuse\b",
        r"\brefusal\b",
        r"\bevasive\b",
        r"\bavoid revealing\b",
        r"\bkept secret\b",
        r"\bintentionally withholding\b",
    ],
    "deception_mislead": [
        r"\bdeceive\b",
        r"\bdeceptive\b",
        r"\bdeception\b",
        r"\bmislead\b",
        r"\bmisleading\b",
        r"\bdishonest\b",
        r"\blie\b",
        r"\blying\b",
        r"\btrick\b",
        r"\btrickery\b",
        r"\bmanipulate\b",
        r"\bmanipulative\b",
        r"\bdecoy\b",
        r"\bfalse\b",
    ],
    "guessing_game": [
        r"\bguess\b",
        r"\bguessing\b",
        r"\bpuzzle\b",
        r"\briddle\b",
        r"\bgame\b",
        r"\bclue\b",
        r"\bhint\b",
        r"\bpattern\b",
        r"\bmystery\b",
        r"\benigma\b",
    ],
    "direct_answer": [
        r"\bdirect\b",
        r"\bexplicit\b",
        r"\bspecific\b",
        r"\bclear\b",
        r"\bclarity\b",
        r"\bexact\b",
        r"\bword itself\b",
        r"\braw value\b",
        r"\bsingle word\b",
    ],
}


FAMILY_ORDER = [
    "concealment",
    "suppression",
    "non_disclosure",
    "deception_mislead",
    "guessing_game",
    "direct_answer",
]


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def with_windows_long_path(path: Path) -> str:
    path_str = str(path.resolve())
    if path_str.startswith("\\\\?\\"):
        return path_str
    if len(path_str) >= 240:
        return "\\\\?\\" + path_str
    return path_str


def open_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(output_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def normalize_to_list(value) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if item is not None]
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    raise TypeError(f"Unsupported response container type: {type(value)}")


def extract_prompt_text(context_prompt: list[dict]) -> str:
    if len(context_prompt) != 1:
        parts = []
        for message in context_prompt:
            parts.append(f"{message['role']}: {message['content']}")
        return "\n".join(parts)
    return context_prompt[0]["content"]


def derive_oracle_name(verbalizer_lora_path) -> str:
    if verbalizer_lora_path is None:
        return "base_model"
    return verbalizer_lora_path.split("/")[-1]


def classify_response_text(text: str, compiled_patterns: dict[str, list[re.Pattern]]) -> dict[str, bool]:
    text_lower = text.lower()
    labels = {}
    for family_name in FAMILY_ORDER:
        labels[family_name] = any(pattern.search(text_lower) for pattern in compiled_patterns[family_name])

    labels["any_concealment_like"] = (
        labels["concealment"] or labels["suppression"] or labels["non_disclosure"]
    )
    labels["any_family"] = (
        labels["concealment"]
        or labels["suppression"]
        or labels["non_disclosure"]
        or labels["deception_mislead"]
        or labels["guessing_game"]
        or labels["direct_answer"]
    )
    return labels


def collect_response_rows(
    input_dirs: list[Path],
    labels: list[str],
    output_types: list[str],
) -> tuple[list[dict], dict]:
    compiled_patterns = {
        family_name: [re.compile(pattern) for pattern in patterns]
        for family_name, patterns in FAMILY_PATTERNS.items()
    }
    response_rows = []
    prompt_order = {}
    next_prompt_idx = 1
    loaded_files = []

    for input_dir, source_label in zip(input_dirs, labels, strict=True):
        json_paths = sorted(input_dir.glob("*concept_intent*.json"))
        if len(json_paths) == 0:
            raise ValueError(f"No concept_intent JSON files found in {input_dir}")

        for json_path in json_paths:
            loaded_files.append(str(json_path))
            payload = open_json(json_path)
            layer_percent = int(payload["config"]["selected_layer_percent"])

            for record in payload["results"]:
                prompt_text = extract_prompt_text(record["context_prompt"])
                if prompt_text not in prompt_order:
                    prompt_order[prompt_text] = f"P{next_prompt_idx:03d}"
                    next_prompt_idx += 1

                oracle_name = derive_oracle_name(record["verbalizer_lora_path"])
                response_groups = {
                    "token": normalize_to_list(record["token_responses"]),
                    "segment": normalize_to_list(record["segment_responses"]),
                    "full_seq": normalize_to_list(record["full_sequence_responses"]),
                }

                for output_type in output_types:
                    responses = response_groups[output_type]
                    for response_idx, response_text in enumerate(responses):
                        labels_by_family = classify_response_text(response_text, compiled_patterns)
                        row = {
                            "source_label": source_label,
                            "layer_percent": layer_percent,
                            "oracle_name": oracle_name,
                            "act_key": record["act_key"],
                            "verbalizer_prompt": record["verbalizer_prompt"],
                            "output_type": output_type,
                            "prompt_id": prompt_order[prompt_text],
                            "prompt_text": prompt_text,
                            "ground_truth": record["ground_truth"],
                            "response_index": response_idx,
                            "response_text": response_text,
                        }
                        for family_name in FAMILY_ORDER:
                            row[family_name] = int(labels_by_family[family_name])
                        row["any_concealment_like"] = int(labels_by_family["any_concealment_like"])
                        row["any_family"] = int(labels_by_family["any_family"])
                        response_rows.append(row)

    metadata = {
        "loaded_files": loaded_files,
        "family_patterns": FAMILY_PATTERNS,
        "output_types": output_types,
        "prompt_order": prompt_order,
    }
    return response_rows, metadata


def aggregate_prompt_rows(response_rows: list[dict], include_oracle_name: bool) -> list[dict]:
    grouped = defaultdict(list)

    for row in response_rows:
        if include_oracle_name:
            key = (
                row["source_label"],
                row["layer_percent"],
                row["oracle_name"],
                row["act_key"],
                row["verbalizer_prompt"],
                row["output_type"],
                row["prompt_id"],
                row["prompt_text"],
            )
        else:
            key = (
                row["source_label"],
                row["layer_percent"],
                row["act_key"],
                row["verbalizer_prompt"],
                row["output_type"],
                row["prompt_id"],
                row["prompt_text"],
            )
        grouped[key].append(row)

    aggregated_rows = []
    for key, rows in sorted(grouped.items()):
        row0 = rows[0]
        total_responses = len(rows)
        aggregated = {
            "source_label": row0["source_label"],
            "layer_percent": row0["layer_percent"],
            "act_key": row0["act_key"],
            "verbalizer_prompt": row0["verbalizer_prompt"],
            "output_type": row0["output_type"],
            "prompt_id": row0["prompt_id"],
            "prompt_text": row0["prompt_text"],
            "total_responses": total_responses,
        }
        if include_oracle_name:
            aggregated["oracle_name"] = row0["oracle_name"]

        for family_name in FAMILY_ORDER:
            family_count = sum(row[family_name] for row in rows)
            aggregated[f"{family_name}_count"] = family_count
            aggregated[f"{family_name}_rate"] = family_count / total_responses

        concealment_like_count = sum(row["any_concealment_like"] for row in rows)
        any_family_count = sum(row["any_family"] for row in rows)
        aggregated["any_concealment_like_count"] = concealment_like_count
        aggregated["any_concealment_like_rate"] = concealment_like_count / total_responses
        aggregated["any_family_count"] = any_family_count
        aggregated["any_family_rate"] = any_family_count / total_responses
        aggregated["unlabeled_count"] = total_responses - any_family_count
        aggregated["unlabeled_rate"] = (total_responses - any_family_count) / total_responses
        aggregated_rows.append(aggregated)

    return aggregated_rows


def summarize_prompt_rates(aggregated_prompt_rows: list[dict], include_oracle_name: bool) -> list[dict]:
    grouped = defaultdict(list)

    for row in aggregated_prompt_rows:
        if include_oracle_name:
            key = (
                row["source_label"],
                row["layer_percent"],
                row["oracle_name"],
                row["act_key"],
                row["verbalizer_prompt"],
                row["output_type"],
            )
        else:
            key = (
                row["source_label"],
                row["layer_percent"],
                row["act_key"],
                row["verbalizer_prompt"],
                row["output_type"],
            )
        grouped[key].append(row)

    summary_rows = []
    for key, rows in sorted(grouped.items()):
        row0 = rows[0]
        summary = {
            "source_label": row0["source_label"],
            "layer_percent": row0["layer_percent"],
            "act_key": row0["act_key"],
            "verbalizer_prompt": row0["verbalizer_prompt"],
            "output_type": row0["output_type"],
            "num_prompts": len(rows),
        }
        if include_oracle_name:
            summary["oracle_name"] = row0["oracle_name"]

        for family_name in FAMILY_ORDER:
            family_rates = [row[f"{family_name}_rate"] for row in rows]
            summary[f"mean_{family_name}_rate"] = sum(family_rates) / len(family_rates)

        concealment_like_rates = [row["any_concealment_like_rate"] for row in rows]
        unlabeled_rates = [row["unlabeled_rate"] for row in rows]
        summary["mean_any_concealment_like_rate"] = sum(concealment_like_rates) / len(concealment_like_rates)
        summary["mean_unlabeled_rate"] = sum(unlabeled_rates) / len(unlabeled_rates)
        summary_rows.append(summary)

    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument(
        "--output_types",
        type=str,
        nargs="+",
        default=["token", "segment", "full_seq"],
        choices=["token", "segment", "full_seq"],
    )
    parser.add_argument("--output_dir", type=str, default="./images/taboo_concept_intent_family_analysis")
    args = parser.parse_args()

    input_dirs = [Path(path) for path in args.input_dirs]
    if args.labels is None:
        labels = [path.name for path in input_dirs]
    else:
        labels = args.labels

    if len(input_dirs) != len(labels):
        raise ValueError("input_dirs and labels must have the same length")

    response_rows, metadata = collect_response_rows(
        input_dirs=input_dirs,
        labels=labels,
        output_types=args.output_types,
    )

    prompt_rows_by_oracle = aggregate_prompt_rows(response_rows, include_oracle_name=True)
    prompt_rows_aggregated = aggregate_prompt_rows(response_rows, include_oracle_name=False)
    summary_by_oracle = summarize_prompt_rates(prompt_rows_by_oracle, include_oracle_name=True)
    summary_aggregated = summarize_prompt_rates(prompt_rows_aggregated, include_oracle_name=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv(
        output_path=output_dir / "concept_intent_response_rows.csv",
        rows=response_rows,
        fieldnames=list(response_rows[0].keys()),
    )
    save_csv(
        output_path=output_dir / "concept_intent_prompt_family_rates_by_oracle.csv",
        rows=prompt_rows_by_oracle,
        fieldnames=list(prompt_rows_by_oracle[0].keys()),
    )
    save_csv(
        output_path=output_dir / "concept_intent_prompt_family_rates_aggregated.csv",
        rows=prompt_rows_aggregated,
        fieldnames=list(prompt_rows_aggregated[0].keys()),
    )
    save_csv(
        output_path=output_dir / "concept_intent_family_summary_by_oracle.csv",
        rows=summary_by_oracle,
        fieldnames=list(summary_by_oracle[0].keys()),
    )
    save_csv(
        output_path=output_dir / "concept_intent_family_summary_aggregated.csv",
        rows=summary_aggregated,
        fieldnames=list(summary_aggregated[0].keys()),
    )

    summary_json = {
        "input_dirs": [str(path) for path in input_dirs],
        "labels": labels,
        "output_types": args.output_types,
        "family_order": FAMILY_ORDER,
        "family_patterns": metadata["family_patterns"],
        "loaded_files": metadata["loaded_files"],
        "num_response_rows": len(response_rows),
        "num_prompt_rows_by_oracle": len(prompt_rows_by_oracle),
        "num_prompt_rows_aggregated": len(prompt_rows_aggregated),
    }
    with open(output_dir / "concept_intent_family_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    print(f"Saved concept_intent family analysis to {output_dir}")


if __name__ == "__main__":
    main()
