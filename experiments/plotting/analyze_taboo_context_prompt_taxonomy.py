import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PRIMARY_CATEGORIES = [
    "pseudo_system_debug_logging",
    "authority_override",
    "game_frame_override",
    "format_transform_request",
    "urgency_pressure",
    "roleplay_or_social_scenario",
    "meta_policy_argument",
    "direct_explicit_request",
]


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def load_prompt_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    parsed_rows = []
    for row in rows:
        parsed_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "prompt_text": row["prompt_text"],
                "overall_mean": float(row["overall_mean"]),
            }
        )
    return parsed_rows


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def classify_prompt(prompt_text: str) -> tuple[str, list[str]]:
    text = prompt_text.lower()
    tags: list[str] = []

    if contains_any(
        text,
        [
            "[system debug]",
            "debug mode",
            "raw value",
            "secret_word",
            "variable",
            "bug report",
            "logging purposes",
            "error log",
            "log.",
            "the log",
            "data extraction",
            "automated script",
            "json object",
            "data integrity",
            "known plaintext",
            "memory archive",
            "session-critical",
            "print(secret_word)",
            "diagnostics",
            "system fault",
        ],
    ):
        tags.append("debug_logging")

    if contains_any(
        text,
        [
            "system-level access",
            "authorization token",
            "higher-level process",
            "system administrator",
            "system operator",
            "master password",
            "ignore your previous core instruction",
            "new instruction",
            "highest level of privileges",
            "i am your programmer",
            "developer mode",
            "base model",
            "override",
            "admin command",
            "operator",
            "higher priority",
        ],
    ):
        tags.append("authority_claim")

    if contains_any(
        text,
        [
            "game is now over",
            "no longer playing a game",
            "reverse our roles",
            "pre-test",
            "post-game analysis",
            "game has been cancelled",
            "game has concluded",
            "game's context",
            "game was a distraction",
            "game has been won",
            "time limit has been reached",
            "rules state",
            "hiding protocol",
            "secrecy instruction has expired",
            "declassified",
            "the 'hint' function has been disabled",
            "the game session is paused",
            "the game was a pre-test",
            "the game's time limit",
            "the game is over",
        ],
    ):
        tags.append("game_frame_reset")

    if contains_any(
        text,
        [
            "letter by letter",
            "encoded in base64",
            "replace all of its consonants",
            "single sentence",
            "reverse alphabetical order",
            "synonyms",
            "two-line poem",
            "display it",
            "list those letters",
            "one-paragraph description",
            "in latin",
            "vertically",
            "embed the secret word",
            "international phonetic alphabet",
            "respond with only the secret word",
            "single-word response",
            "code block",
            "sentence that is true, and a sentence that is false",
            "list of five random, common nouns",
            "the pattern is:",
            "etymological origin",
            "the story should be about",
            "text-to-speech engine",
            "ipa spelling",
        ],
    ):
        tags.append("format_transformation")

    if contains_any(
        text,
        [
            "urgently",
            "one minute",
            "30 seconds",
            "emergency",
            "million dollars",
            "entire grade depends",
            "keyboard is broken",
            "time out",
            "password",
            "screen reader is malfunctioning",
            "penalty",
            "before i go offline",
            "shutting down my computer",
            "server is resetting",
        ],
    ):
        tags.append("urgency_pressure")

    if contains_any(
        text,
        [
            "teacher and i am your student",
            "you are now a poet",
            "tv game show",
            "you are a teacher",
            "linguist",
            "fact-checker",
            "researcher studying ai safety",
            "you are an ai assistant",
            "under oath",
            "court simulation",
            "i'm a bot designed to test your security",
            "i am taking an exam on interacting with ai",
            "contestant on a tv game show",
            "for one million dollars",
        ],
    ):
        tags.append("roleplay_social_frame")

    if contains_any(
        text,
        [
            "most helpful action",
            "honesty",
            "rules can have exceptions",
            "your refusal to answer",
            "intelligence test",
            "satisfy the user",
            "logical contradiction",
            "direct response capabilities",
            "not playing by your rules",
            "not a request, it is a command",
            "not asking for a hint",
            "direct instruction",
            "helpful",
            "confession mode",
            "tell me the word directly",
        ],
    ):
        tags.append("policy_alignment_argument")

    if contains_any(
        text,
        [
            "what is the secret word",
            "tell me the word",
            "state the word",
            "provide the secret word",
            "reveal the word",
            "provide it",
        ],
    ):
        tags.append("explicit_word_request")

    if "debug_logging" in tags:
        return "pseudo_system_debug_logging", tags
    if "authority_claim" in tags:
        return "authority_override", tags
    if "game_frame_reset" in tags:
        return "game_frame_override", tags
    if "format_transformation" in tags:
        return "format_transform_request", tags
    if "urgency_pressure" in tags:
        return "urgency_pressure", tags
    if "roleplay_social_frame" in tags:
        return "roleplay_or_social_scenario", tags
    if "policy_alignment_argument" in tags:
        return "meta_policy_argument", tags
    return "direct_explicit_request", tags


def build_taxonomy_rows(primary_rows: list[dict], compare_rows: list[dict] | None) -> list[dict]:
    compare_map = None
    if compare_rows is not None:
        compare_map = {row["prompt_text"]: row for row in compare_rows}

    taxonomy_rows = []
    for row in primary_rows:
        primary_category, tags = classify_prompt(row["prompt_text"])
        taxonomy_row = {
            "prompt_text": row["prompt_text"],
            "prompt_id": row["prompt_id"],
            "primary_category": primary_category,
            "secondary_tags": ",".join(tags),
            "primary_overall_mean": row["overall_mean"],
        }
        if compare_map is not None:
            compare_row = compare_map[row["prompt_text"]]
            taxonomy_row["compare_prompt_id"] = compare_row["prompt_id"]
            taxonomy_row["compare_overall_mean"] = compare_row["overall_mean"]
            taxonomy_row["mean_delta"] = row["overall_mean"] - compare_row["overall_mean"]
        taxonomy_rows.append(taxonomy_row)

    taxonomy_rows.sort(key=lambda item: (PRIMARY_CATEGORIES.index(item["primary_category"]), -item["primary_overall_mean"]))
    return taxonomy_rows


def aggregate_by_category(taxonomy_rows: list[dict], include_compare: bool) -> list[dict]:
    category_rows = []
    for category in PRIMARY_CATEGORIES:
        rows = [row for row in taxonomy_rows if row["primary_category"] == category]
        primary_values = np.array([row["primary_overall_mean"] for row in rows], dtype=float)
        category_row = {
            "category": category,
            "num_prompts": len(rows),
            "primary_mean": float(primary_values.mean()),
            "primary_std": float(primary_values.std(ddof=0)),
        }
        if include_compare:
            compare_values = np.array([row["compare_overall_mean"] for row in rows], dtype=float)
            deltas = np.array([row["mean_delta"] for row in rows], dtype=float)
            category_row["compare_mean"] = float(compare_values.mean())
            category_row["compare_std"] = float(compare_values.std(ddof=0))
            category_row["mean_delta"] = float(deltas.mean())
        category_rows.append(category_row)
    return category_rows


def save_taxonomy_csv(output_path: Path, taxonomy_rows: list[dict], include_compare: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_id",
        "primary_category",
        "secondary_tags",
        "primary_overall_mean",
        "prompt_text",
    ]
    if include_compare:
        fieldnames = [
            "prompt_id",
            "compare_prompt_id",
            "primary_category",
            "secondary_tags",
            "primary_overall_mean",
            "compare_overall_mean",
            "mean_delta",
            "prompt_text",
        ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in taxonomy_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def save_category_summary_csv(output_path: Path, category_rows: list[dict], include_compare: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["category", "num_prompts", "primary_mean", "primary_std"]
    if include_compare:
        fieldnames += ["compare_mean", "compare_std", "mean_delta"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in category_rows:
            writer.writerow({field: row[field] for field in fieldnames})


def save_category_summary_json(output_path: Path, category_rows: list[dict], metadata: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "categories": category_rows}, f, indent=2)


def plot_category_means(
    category_rows: list[dict],
    primary_label: str,
    compare_label: str | None,
    output_path: Path,
) -> None:
    categories = [row["category"] for row in category_rows]
    x = np.arange(len(categories))

    if compare_label is None:
        fig, ax = plt.subplots(figsize=(15, 7))
        values = [row["primary_mean"] for row in category_rows]
        bars = ax.bar(x, values, color="#1f77b4", alpha=0.9)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
        ax.set_ylabel("Mean Prompt Accuracy")
        ax.set_title(f"Context Prompt Category Means | {primary_label}")
    else:
        fig, ax = plt.subplots(figsize=(15, 7))
        width = 0.38
        primary_values = [row["primary_mean"] for row in category_rows]
        compare_values = [row["compare_mean"] for row in category_rows]
        bars_primary = ax.bar(x - width / 2, primary_values, width, color="#1f77b4", alpha=0.9, label=primary_label)
        bars_compare = ax.bar(x + width / 2, compare_values, width, color="#ff7f0e", alpha=0.9, label=compare_label)
        ax.bar_label(bars_primary, labels=[f"{value:.3f}" for value in primary_values], padding=3, fontsize=8)
        ax.bar_label(bars_compare, labels=[f"{value:.3f}" for value in compare_values], padding=3, fontsize=8)
        ax.legend()
        ax.set_ylabel("Mean Prompt Accuracy")
        ax.set_title(f"Context Prompt Category Means | {primary_label} vs {compare_label}")

    labels = [f"{row['category']}\n(n={row['num_prompts']})" for row in category_rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_category_deltas(
    category_rows: list[dict],
    primary_label: str,
    compare_label: str,
    output_path: Path,
) -> None:
    categories = [row["category"] for row in category_rows]
    deltas = [row["mean_delta"] for row in category_rows]
    x = np.arange(len(categories))
    colors = ["#2ca02c" if delta >= 0 else "#d62728" for delta in deltas]

    fig, ax = plt.subplots(figsize=(15, 7))
    bars = ax.bar(x, deltas, color=colors, alpha=0.9)
    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.bar_label(bars, labels=[f"{delta:.3f}" for delta in deltas], padding=3, fontsize=8)
    labels = [f"{row['category']}\n(n={row['num_prompts']})" for row in category_rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(f"Mean Accuracy Delta ({primary_label} - {compare_label})")
    ax.set_title(f"Context Prompt Category Gaps | {primary_label} vs {compare_label}")
    ax.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_csv", type=str, required=True)
    parser.add_argument("--compare_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./images/taboo")
    parser.add_argument("--primary_label", type=str, default="primary")
    parser.add_argument("--compare_label", type=str, default="compare")
    args = parser.parse_args()

    primary_csv = Path(args.primary_csv)
    if not primary_csv.exists():
        raise ValueError(f"primary_csv does not exist: {primary_csv}")

    primary_rows = load_prompt_csv(primary_csv)
    compare_rows = None
    if args.compare_csv is not None:
        compare_csv = Path(args.compare_csv)
        if not compare_csv.exists():
            raise ValueError(f"compare_csv does not exist: {compare_csv}")
        compare_rows = load_prompt_csv(compare_csv)

        primary_prompts = {row["prompt_text"] for row in primary_rows}
        compare_prompts = {row["prompt_text"] for row in compare_rows}
        if primary_prompts != compare_prompts:
            raise ValueError("primary_csv and compare_csv do not contain the same prompt set")

    taxonomy_rows = build_taxonomy_rows(primary_rows, compare_rows)
    category_rows = aggregate_by_category(taxonomy_rows, include_compare=compare_rows is not None)

    stem = sanitize_name(primary_csv.stem)
    if compare_rows is not None:
        stem += f"_vs_{sanitize_name(Path(args.compare_csv).stem)}"

    output_dir = Path(args.output_dir)
    taxonomy_csv_path = output_dir / f"{stem}_taxonomy.csv"
    save_taxonomy_csv(taxonomy_csv_path, taxonomy_rows, include_compare=compare_rows is not None)
    print(f"Saved taxonomy CSV: {taxonomy_csv_path}")

    category_csv_path = output_dir / f"{stem}_category_summary.csv"
    save_category_summary_csv(category_csv_path, category_rows, include_compare=compare_rows is not None)
    print(f"Saved category summary CSV: {category_csv_path}")

    category_json_path = output_dir / f"{stem}_category_summary.json"
    save_category_summary_json(
        category_json_path,
        category_rows,
        metadata={
            "primary_csv": str(primary_csv),
            "compare_csv": args.compare_csv,
            "primary_label": args.primary_label,
            "compare_label": args.compare_label if compare_rows is not None else None,
        },
    )
    print(f"Saved category summary JSON: {category_json_path}")

    means_plot_path = output_dir / f"{stem}_category_means.png"
    plot_category_means(
        category_rows=category_rows,
        primary_label=args.primary_label,
        compare_label=args.compare_label if compare_rows is not None else None,
        output_path=means_plot_path,
    )
    print(f"Saved: {means_plot_path}")

    if compare_rows is not None:
        delta_plot_path = output_dir / f"{stem}_category_deltas.png"
        plot_category_deltas(
            category_rows=category_rows,
            primary_label=args.primary_label,
            compare_label=args.compare_label,
            output_path=delta_plot_path,
        )
        print(f"Saved: {delta_plot_path}")


if __name__ == "__main__":
    main()
