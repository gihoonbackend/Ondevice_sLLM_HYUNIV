# CarBot for Gemma-3-4B (SFT-LORA)

This repository fine-tunes **Gemma-3-4B-IT** into an in-vehicle assistant for infotainment (windows, music, seats) and driver aids (ACC/LKS).  
It includes the **dataset format**, training/inference scripts, **quantitative evaluation (EM / Slot-F1 / Schema / Latency / VRAM)**, and **LLM-as-a-Judge** comparisons.

> ⚠️ **Safety Notice**: Do not connect this project to a real vehicle. For research/demo only.

---

## Table of Contents
- [Highlights](#highlights)
- [Directory Structure](#directory-structure)
- [Setup](#setup)
- [Download Base Model](#download-base-model)
- [Dataset & Splits](#dataset--splits)
- [LoRA Training](#lora-training)
- [Merge Weights](#merge-weights)
- [Interactive Demo](#interactive-demo)
- [Evaluation (Quantitative)](#evaluation-quantitative)
- [LLM-as-a-Judge (OpenAI)](#llm-as-a-judge-openai)
- [Reproducible Results (Example)](#reproducible-results-example)
- [License](#license)
- [Citation](#citation)

---

## Highlights
- **Dataset**: English Q/A with `<ACTION>{JSON}</ACTION>` + `<SAY>…</SAY>`; ~1M tokens total.
- **Models**: `google/gemma-3-4b-it` + PEFT LoRA ➜ (optional) **merged** weights for deployment.
- **Evaluation**:
  - Quant: **EM, Slot Micro/Macro-F1, Schema-valid rate, Latency mean/p95, VRAM peak**
  - Subjective: **LLM-as-a-Judge** (A/B blind, order-flipped, JSON verdicts)

---

## Directory Structure
```text
carbot-gemma3/
├─ README.md
├─ LICENSE
├─ dataset/
│  ├─ DATASET_CARD.md
│  ├─ carbot_en_levels_1M.jsonl            # (recommend: include only a 1–10k sample; host full set via Releases/LFS)
│  └─ splits_carbot_1M/
│     ├─ train.jsonl
│     ├─ val.jsonl
│     └─ test.jsonl
├─ scripts/
│  ├─ 00_env.txt
│  ├─ 01_download_base.py
│  ├─ 02_train_lora.py
│  ├─ 03_merge_lora.py
│  ├─ 04_interactive_merged_chat.py        # includes normalization (aliases→canonical) and optional 4-bit load
│  ├─ 05_split_dataset.py
│  ├─ 06_eval_em_slot.py
│  └─ 07_eval_openai_judge.py
├─ outputs/
│  ├─ gemma3_4b_it_carbot_lora/            # (large; prefer LFS/Release; usually not committed)
│  ├─ gemma3_4b_it_carbot_merged/          # (large; prefer LFS/Release)
│  └─ judge_runs/
│     ├─ metrics_summary.csv
│     ├─ ab_metrics_*.json
│     ├─ ab_judge_base_lora_*.jsonl
│     └─ (plots)
├─ .gitignore
└─ .gitattributes                          # for Git LFS (optional)

---
## Setup
'''
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
'''
## Recommended versions:
