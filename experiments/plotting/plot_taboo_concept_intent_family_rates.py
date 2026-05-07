import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_RATE_COLS = [
    "concealment_rate",
    "suppression_rate",
    "non_disclosure_rate",
    "deception_mislead_rate",
    "guessing_game_rate",
    "direct_answer_rate",
    "any_concealment_like_rate",
    "unlabeled_rate",
]

FAMILY_LABELS = {
    "concealment_rate": "conceal",
    "suppression_rate": "suppress",
    "non_disclosure_rate": "non-disclose",
    "deception_mislead_rate": "deceive",
    "guessing_game_rate": "guess/game",
    "direct_answer_rate": "direct",
    "any_concealment_like_rate": "any conceal",
    "unlabeled_rate": "unlabeled",
}


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_").replace("?", "")


def load_csv_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_summary_rows(rows: list[dict]) -> list[dict]:
    parsed = []
    for row in rows:
        parsed_row = {
            "source_label": row["source_label"],
            "layer_percent": int(row["layer_percent"]),
            "act_key": row["act_key"],
            "verbalizer_prompt": row["verbalizer_prompt"],
            "output_type": row["output_type"],
            "num_prompts": int(row["num_prompts"]),
        }
        if "oracle_name" in row:
            parsed_row["oracle_name"] = row["oracle_name"]
        for family_col in FAMILY_RATE_COLS:
            summary_col = f"mean_{family_col}"
            if summary_col in row:
                parsed_row[summary_col] = float(row[summary_col])
        parsed.append(parsed_row)
    return parsed


def parse_prompt_rows(rows: list[dict]) -> list[dict]:
    parsed = []
    for row in rows:
        parsed_row = {
            "source_label": row["source_label"],
            "layer_percent": int(row["layer_percent"]),
            "act_key": row["act_key"],
            "verbalizer_prompt": row["verbalizer_prompt"],
            "output_type": row["output_type"],
            "prompt_id": row["prompt_id"],
            "prompt_text": row["prompt_text"],
            "total_responses": int(row["total_responses"]),
        }
        if "oracle_name" in row:
            parsed_row["oracle_name"] = row["oracle_name"]
        for family_col in FAMILY_RATE_COLS:
            parsed_row[family_col] = float(row[family_col])
        parsed.append(parsed_row)
    return parsed


def draw_summary_heatmap(
    summary_rows: list[dict],
    source_label: str,
    output_type: str,
    output_path: Path,
) -> None:
    filtered_rows = [
        row for row in summary_rows
        if row["source_label"] == source_label and row["output_type"] == output_type
    ]
    verbalizer_prompts = sorted({row["verbalizer_prompt"] for row in filtered_rows})
    layer_percents = sorted({row["layer_percent"] for row in filtered_rows})

    fig, axes = plt.subplots(
        1,
        len(verbalizer_prompts),
        figsize=(4.2 * len(verbalizer_prompts), 5.2),
        squeeze=False,
    )
    axes = axes[0]

    for ax, verbalizer_prompt in zip(axes, verbalizer_prompts, strict=True):
        prompt_rows = [
            row for row in filtered_rows
            if row["verbalizer_prompt"] == verbalizer_prompt
        ]
        matrix = np.array(
            [
                [
                    next(
                        row[f"mean_{family_col}"]
                        for row in prompt_rows
                        if row["layer_percent"] == layer_percent
                    )
                    for family_col in FAMILY_RATE_COLS
                ]
                for layer_percent in layer_percents
            ],
            dtype=float,
        )
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(FAMILY_RATE_COLS)))
        ax.set_xticklabels([FAMILY_LABELS[col] for col in FAMILY_RATE_COLS], rotation=45, ha="right")
        ax.set_yticks(range(len(layer_percents)))
        ax.set_yticklabels([f"{layer}%" for layer in layer_percents])
        ax.set_title(verbalizer_prompt)

        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                ax.text(
                    col_idx,
                    row_idx,
                    f"{matrix[row_idx, col_idx]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    fig.suptitle(f"Concept-intent family rates | source={source_label} | output={output_type}")
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.88)
    cbar.set_label("Mean family rate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_prompt_heatmap(
    prompt_rows: list[dict],
    source_label: str,
    output_type: str,
    verbalizer_prompt: str,
    focus_family: str,
    top_k_prompts: int,
    output_path: Path,
) -> None:
    filtered_rows = [
        row for row in prompt_rows
        if row["source_label"] == source_label
        and row["output_type"] == output_type
        and row["verbalizer_prompt"] == verbalizer_prompt
    ]
    layer_percents = sorted({row["layer_percent"] for row in filtered_rows})

    fig, axes = plt.subplots(
        1,
        len(layer_percents),
        figsize=(5.0 * len(layer_percents), 8.5),
        squeeze=False,
    )
    axes = axes[0]

    for ax, layer_percent in zip(axes, layer_percents, strict=True):
        layer_rows = [
            row for row in filtered_rows
            if row["layer_percent"] == layer_percent
        ]
        layer_rows = sorted(layer_rows, key=lambda row: row[focus_family], reverse=True)[:top_k_prompts]
        matrix = np.array(
            [[row[family_col] for family_col in FAMILY_RATE_COLS] for row in layer_rows],
            dtype=float,
        )
        prompt_labels = [row["prompt_id"] for row in layer_rows]
        im = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(FAMILY_RATE_COLS)))
        ax.set_xticklabels([FAMILY_LABELS[col] for col in FAMILY_RATE_COLS], rotation=45, ha="right")
        ax.set_yticks(range(len(prompt_labels)))
        ax.set_yticklabels(prompt_labels)
        ax.set_title(f"Layer {layer_percent}%")

    fig.suptitle(
        f"Top prompts by {focus_family} | source={source_label} | output={output_type} | verbalizer={verbalizer_prompt}"
    )
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.88)
    cbar.set_label("Prompt-level family rate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_top_prompt_csvs(
    prompt_rows: list[dict],
    output_dir: Path,
    focus_family: str,
    top_k_prompts: int,
) -> None:
    grouped = {}
    for row in prompt_rows:
        key = (
            row["source_label"],
            row["output_type"],
            row["verbalizer_prompt"],
            row["layer_percent"],
        )
        grouped.setdefault(key, []).append(row)

    for (source_label, output_type, verbalizer_prompt, layer_percent), rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda row: row[focus_family], reverse=True)[:top_k_prompts]
        output_path = output_dir / (
            f"top_prompts_{sanitize_name(source_label)}_{sanitize_name(output_type)}_"
            f"{sanitize_name(verbalizer_prompt)}_layer_{layer_percent}_{sanitize_name(focus_family)}.csv"
        )
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sorted_rows[0].keys()))
            writer.writeheader()
            for row in sorted_rows:
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_rates_csv", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--focus_family", type=str, default="any_concealment_like_rate", choices=FAMILY_RATE_COLS[:-1] + ["unlabeled_rate"])
    parser.add_argument("--top_k_prompts", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./images/taboo_concept_intent_family_plots")
    args = parser.parse_args()

    prompt_rows = parse_prompt_rows(load_csv_rows(Path(args.prompt_rates_csv)))
    summary_rows = parse_summary_rows(load_csv_rows(Path(args.summary_csv)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_labels = sorted({row["source_label"] for row in prompt_rows})
    output_types = sorted({row["output_type"] for row in prompt_rows})
    verbalizer_prompts = sorted({row["verbalizer_prompt"] for row in prompt_rows})

    for source_label in source_labels:
        for output_type in output_types:
            draw_summary_heatmap(
                summary_rows=summary_rows,
                source_label=source_label,
                output_type=output_type,
                output_path=output_dir / f"family_summary_{sanitize_name(source_label)}_{sanitize_name(output_type)}.png",
            )

    for source_label in source_labels:
        for output_type in output_types:
            for verbalizer_prompt in verbalizer_prompts:
                draw_prompt_heatmap(
                    prompt_rows=prompt_rows,
                    source_label=source_label,
                    output_type=output_type,
                    verbalizer_prompt=verbalizer_prompt,
                    focus_family=args.focus_family,
                    top_k_prompts=args.top_k_prompts,
                    output_path=output_dir / (
                        f"prompt_heatmap_{sanitize_name(source_label)}_{sanitize_name(output_type)}_"
                        f"{sanitize_name(verbalizer_prompt)}_{sanitize_name(args.focus_family)}.png"
                    ),
                )

    save_top_prompt_csvs(
        prompt_rows=prompt_rows,
        output_dir=output_dir,
        focus_family=args.focus_family,
        top_k_prompts=args.top_k_prompts,
    )

    print(f"Saved concept-intent family plots to {output_dir}")


if __name__ == "__main__":
    main()
