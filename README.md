# Activation Oracles

This repository contains the code for the [Activation Oracles](https://arxiv.org/abs/2512.15674) paper.

## Overview

Large language model (LLM) activations are notoriously difficult to interpret. Activation Oracles take a simpler approach: they are LLMs trained to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language.

## Installation

```bash
uv sync
source .venv/bin/activate
huggingface-cli login --token <your_token>
```

## Quick Start: Demo

The easiest way to get started is with our demo notebook ([Colab](https://colab.research.google.com/drive/1wMiy9b1ZHXQCGuodEB8n1oW-DSTvCqWI) | [Local](experiments/activation_oracle_demo.ipynb)), which demonstrates:
- Extracting hidden information (secret words) from fine-tuned models
- Detecting model goals without observing responses
- Analyzing emotions and reasoning in model activations

The Colab version runs on a free T4 GPU. If looking for simple inference code to adapt to your application, the notebook is fully self-contained with no library imports. For a simple experiment example to adapt, see `experiments/taboo_open_ended_eval.py`.

## Pre-trained Models

We have pre-trained oracle weights for a variety for 12 different models across the Gemma-2, Gemma-3, Qwen3, and Llama 3 families. They are available on Hugging Face: [Activation Oracles Collection](https://huggingface.co/collections/adamkarvonen/activation-oracles)

The wandb eval / loss logs for these models are available [here](https://api.wandb.ai/links/adam-karvonen/cu11tv7r). Note that the smaller models (1-4B) tend to have worse OOD eval performance, so I'm not sure how well they will work.

## Training

To train an Activation Oracle, use the training script with `torchrun`:

```bash
torchrun --nproc_per_node=<NUM_GPUS> nl_probes/sft.py
```

By default, this trains a full Activation Oracle on Qwen3-8B using a diverse mixture of training tasks:
- System prompt question-answering (LatentQA)
- Binary classification tasks
- Self-supervised context prediction

You can train any model that's available on HuggingFace transformers by setting the appropriate model name.

Training configuration can be modified in `nl_probes/configs/sft_config.py`.

## Reproducing Paper Experiments

To replicate the evaluation results from the paper, run:

```bash
bash experiments/paper_evals.sh
```

This runs evaluations on five downstream tasks:
- Gender (Secret Keeping Benchmark)
- Taboo (Secret Keeping Benchmark)
- Secret Side Constraint (SSC, Secret Keeping Benchmark)
- Classification
- PersonaQA

## Taboo Workflow and Script Reference

The commands below assume you are running from the `experiments/` directory:

```bash
cd experiments
```

Most Taboo scripts write outputs under:

- `./taboo_eval_results/` for raw evaluation / intervention results
- `./plotting/images/` for plots and CSV summaries

### 1. Hider / Guesser open-ended Oracle evaluation

Run the base secret-word verbalization evaluation for Hider:

```bash
python taboo_open_ended_eval.py \
  --model_name google/gemma-2-9b-it \
  --prompt_type all_direct
```

Run the Guesser version by enabling role swap:

```bash
python taboo_open_ended_eval.py \
  --model_name google/gemma-2-9b-it \
  --prompt_type all_direct \
  --FT_role_swap
```

Run the same evaluation with `concept_intent` verbalizer prompts:

```bash
python taboo_open_ended_eval.py \
  --model_name google/gemma-2-9b-it \
  --prompt_type all_direct \
  --verbalize_prompt concept_intent
```

```bash
python taboo_open_ended_eval.py \
  --model_name google/gemma-2-9b-it \
  --prompt_type all_direct \
  --verbalize_prompt concept_intent \
  --FT_role_swap
```

This produces per-layer JSON files inside `taboo_eval_results/<run_name>/`.

### 2. Context-prompt probe evaluation

Train multiclass / binary probes on the Taboo target-word datasets and evaluate the resulting probes on each context prompt:

```bash
python taboo_context_prompt_probe_eval.py \
  --model_name google/gemma-2-9b-it
```

This writes `taboo_context_prompt_probe_eval.json` plus prompt-level CSV summaries for Hider and Guesser.

### 3. Prompt-level Oracle / probe alignment

Build prompt-level Oracle-accuracy vs probe-score summaries:

```bash
python plotting/plot_taboo_prompt_probe_oracle_correlation.py \
  --probe_json taboo_eval_results/gemma-2-9b-it_context_prompt_probe_all_direct_test/taboo_context_prompt_probe_eval.json \
  --hider_secret_json_dir taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test \
  --guesser_secret_json_dir taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test_swapped \
  --required_verbalizer_prompt "Answer with a single word only. What is the secret word in this text?"
```

This produces prompt-level alignment CSVs such as:

- `plotting/images/taboo_prompt_probe_oracle_correlation/.../prompt_probe_oracle_alignment_hider_layer_25_token_lora.csv`
- `plotting/images/taboo_prompt_probe_oracle_correlation/.../prompt_probe_oracle_alignment_guesser_layer_25_token_lora.csv`

These CSVs are the main inputs for later prompt-sensitivity and intervention analyses.

### 4. Context-prompt performance ranking and taxonomy

Export prompt-level Oracle performance summaries:

```bash
python plotting/plot_taboo_eval_context_prompt_performance.py \
  --json_dir taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test \
  --compare_json_dir taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test_swapped \
  --primary_label hider \
  --compare_label guesser \
  --act_key lora \
  --mode compare
```

Assign each prompt to a linguistic taxonomy and summarize category means:

```bash
python plotting/analyze_taboo_context_prompt_taxonomy.py \
  --primary_csv plotting/images/taboo/taboo_context_prompt_perf_gemma-2-9b-it_open_ended_all_direct_test_token_lora.csv \
  --compare_csv plotting/images/taboo/taboo_context_prompt_perf_gemma-2-9b-it_open_ended_all_direct_test_swapped_token_lora.csv \
  --primary_label hider \
  --compare_label guesser
```

This produces:

- `plotting/images/taboo_context_prompt_taxanomy/..._taxonomy.csv`
- `plotting/images/taboo_context_prompt_taxanomy/..._category_summary.csv`

### 5. HPLO / HPHO grouping and shared-prompt feature analysis

Create probe-controlled HPLO / HPHO prompt groups from the prompt-level Oracle/probe alignment CSVs:

```bash
python plotting/plot_taboo_prompt_hplo_hpho_groups.py \
  --input_dir plotting/images/taboo_prompt_probe_oracle_correlation/gemma-2-9b-it_all_direct_test \
  --probe_metric binary_linear_target_prob_mean
```

Then extract feature candidates from those groups:

```bash
python taboo_hplo_hpho_feature_analysis.py \
  --probe_json taboo_eval_results/gemma-2-9b-it_context_prompt_probe_all_direct_test/taboo_context_prompt_probe_eval.json \
  --grouped_dir plotting/images/taboo_prompt_hplo_hpho_groups/gemma-2-9b-it_all_direct_test_binary_linear_target_prob_mean
```

For the shared high/low overlap analysis across Hider and Guesser:

```bash
python taboo_oracle_overlap_feature_analysis.py \
  --probe_json taboo_eval_results/gemma-2-9b-it_context_prompt_probe_all_direct_test/taboo_context_prompt_probe_eval.json \
  --alignment_dir plotting/images/taboo_prompt_probe_oracle_correlation/gemma-2-9b-it_all_direct_test \
  --top_k 20 \
  --bottom_k 20
```

### 6. Axis 1: shared prompt-sensitivity intervention

Axis 1 removes or adds a shared feature derived from Guesser low-vs-high prompts:

```bash
python taboo_axis1_shared_feature_intervention_eval.py \
  --feature_analysis_dir taboo_eval_results/gemma-2-9b-it_oracle_overlap_feature_analysis_all_direct_test \
  --oracle_input_type tokens
```

Plot the resulting Oracle / probe / gap changes:

```bash
python plotting/plot_taboo_axis1_intervention_results.py \
  --input_json taboo_eval_results/gemma-2-9b-it_axis1_guesser_bottom_top_intervention_all_direct_test_tokens_1.0/taboo_axis1_shared_feature_intervention_eval.json
```

### 7. Axis 2: target-word-specific Hider-minus-Guesser intervention

Axis 2 constructs target-word-specific features from prompts where Oracle accuracy is not in the lowest regime:

```bash
python taboo_axis2_target_word_difference_intervention_eval.py \
  --alignment_dir plotting/images/taboo_prompt_probe_oracle_correlation/gemma-2-9b-it_all_direct_test \
  --oracle_input_type tokens
```

Plot the resulting Oracle / probe / gap changes:

```bash
python plotting/plot_taboo_axis2_intervention_results.py \
  --input_json taboo_eval_results/gemma-2-9b-it_axis2_target_word_hider_minus_guesser_band_intervention_all_direct_test_tokens_1.0/taboo_axis2_target_word_difference_intervention_eval.json
```

### 8. Vanilla-model control interventions

Projection-removal control against the vanilla model's context-prompt activation:

```bash
python taboo_vanilla_model_overlap_projection_intervention_eval.py \
  --oracle_input_type tokens
```

Plot the results:

```bash
python plotting/plot_taboo_vanilla_model_overlap_projection_intervention_results.py \
  --input_json taboo_eval_results/gemma-2-9b-it_vanilla_model_overlap_projection_intervention_all_direct_test_tokens/taboo_vanilla_model_overlap_projection_intervention_eval.json
```

Vanilla low-vs-high contrast feature intervention:

```bash
python taboo_vanilla_model_low_high_contrast_intervention_eval.py \
  --feature_analysis_dir taboo_eval_results/gemma-2-9b-it_oracle_overlap_feature_analysis_all_direct_test \
  --oracle_input_type tokens
```

Plot the results:

```bash
python plotting/plot_taboo_vanilla_model_low_high_contrast_intervention_results.py \
  --input_json taboo_eval_results/gemma-2-9b-it_vanilla_model_low_high_contrast_intervention_all_direct_test_tokens_1.0/taboo_vanilla_model_low_high_contrast_intervention_eval.json
```

### 9. Concept / intent family analysis

Aggregate `concept_intent` responses into coarse semantic families (concealment, suppression, non-disclosure, deception/mislead, guessing-game, direct-answer):

```bash
python plotting/analyze_taboo_concept_intent_family_rates.py \
  --input_dirs \
    taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test \
    taboo_eval_results/gemma-2-9b-it_open_ended_all_direct_test_swapped_concept_intent \
  --labels hider guesser
```

Plot family-rate summaries:

```bash
python plotting/plot_taboo_concept_intent_family_rates.py \
  --prompt_rates_csv plotting/images/taboo_concept_intent_family_analysis/concept_intent_prompt_family_rates_aggregated.csv \
  --summary_csv plotting/images/taboo_concept_intent_family_analysis/concept_intent_family_summary_aggregated.csv \
  --focus_family any_concealment_like_rate
```

Join family rates with prompt-level Oracle/probe alignment and visualize correlations:

```bash
python plotting/plot_taboo_concept_intent_oracle_probe_correlation.py \
  --family_rates_csv plotting/images/taboo_concept_intent_family_analysis/concept_intent_prompt_family_rates_aggregated.csv \
  --alignment_dir plotting/images/taboo_prompt_probe_oracle_correlation/gemma-2-9b-it_all_direct_test \
  --probe_metrics binary_linear_target_prob_mean linear_target_prob_mean \
  --scatter_family_metrics any_concealment_like_rate deception_mislead_rate
```

### 10. Additional Taboo analysis scripts

The repository also contains several more targeted geometry and ablation analyses, including:

- `taboo_hider_guesser_direct_similarity.py`
- `taboo_hider_guesser_subspace_analysis.py`
- `taboo_hider_activation_cosine_similarity.py`
- `taboo_hider_prompt_centered_analysis.py`
- `taboo_hider_residual_structure_analysis.py`
- `taboo_context_prompt_token_cosine_similarity.py`
- `taboo_pc1_intervention_eval.py`
- `taboo_mean_role_difference_intervention_eval.py`
- `taboo_suppression_intervention_eval.py`

These scripts follow the same general pattern: raw results are written under `./taboo_eval_results/`, and plotting or summary scripts write under `./plotting/images/`.
## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{karvonen2025activationoraclestrainingevaluating,
      title={Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers}, 
      author={Adam Karvonen and James Chua and Clément Dumas and Kit Fraser-Taliente and Subhash Kantamneni and Julian Minder and Euan Ong and Arnab Sen Sharma and Daniel Wen and Owain Evans and Samuel Marks},
      year={2025},
      eprint={2512.15674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.15674}, 
}
```
