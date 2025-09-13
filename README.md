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
```
---
## Setup
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
## Recommended versions:
```
accelerate==1.2.1
transformers==4.54.1
trl==0.17.0
peft==0.17.0
datasets>=2.20.0
openai>=1.40.0
bitsandbytes                 # optional (4-bit); may not work on some Linux/GLIBC setups
```
---
## Download Base Model
```
python scripts/01_download_base.py \
  --repo google/gemma-3-4b-it \
  --out  ./models/gemma-3-4b-it
```
---
## Dataset & Splits
```
{
  "messages": [
    {"role":"user","content":"Open the right rear window halfway."},
    {"role":"assistant","content":"<ACTION>{\"name\":\"car.window.set_level\",\"args\":{\"position\":\"rear_right\",\"level\":2}}</ACTION>\n<SAY>Opening the right rear window to level 2.</SAY>"}
  ]
}
```
## Create stratified splits (80/10/10 by intent):
```
python scripts/05_split_dataset.py \
  --in  dataset/carbot_en_levels_1M.jsonl \
  --out dataset/splits_carbot_1M
```

---
## LoRA Training
```
python scripts/02_train_lora.py \
  --base  ./models/gemma-3-4b-it \
  --train dataset/splits_carbot_1M/train.jsonl \
  --val   dataset/splits_carbot_1M/val.jsonl \
  --out   ./outputs/gemma3_4b_it_carbot_lora
```
## Merge Weights
```
python scripts/03_merge_lora.py \
  --base  ./models/gemma-3-4b-it \
  --lora  ./outputs/gemma3_4b_it_carbot_lora \
  --out   ./outputs/gemma3_4b_it_carbot_merged
```
---
## Interactive Demo
```
python scripts/04_interactive_merged_chat.py \
  --model ./outputs/gemma3_4b_it_carbot_merged
```

## I/O example:
```
USER > open the driver window
<ACTION>{"name":"car.window.set_level","args":{"position":"front_left","level":3}}</ACTION>
<SAY>Opening the driver window to level 3.</SAY>
```
---
## Evaluation (Quantitative)
```
python scripts/06_eval_em_slot.py \
  --base ./models/gemma-3-4b-it \
  --lora ./outputs/gemma3_4b_it_carbot_lora \
  --test dataset/splits_carbot_1M/test.jsonl \
  --out  ./outputs/judge_runs
```
Metrics explained

EM: exact JSON match (name and args) vs ground truth

Slot Micro/Macro-F1: partial correctness on args key-value “slots”

Schema-valid rate: allowed action/arg types/ranges obeyed

Latency mean/p95: average & 95th percentile response time

VRAM peak: peak GPU memory during eval

---
## LLM-as-a-Judge (OpenAI)
```
export OPENAI_API_KEY=sk-...
python scripts/07_eval_openai_judge.py \
  --base  ./models/gemma-3-4b-it \
  --lora  ./outputs/gemma3_4b_it_carbot_lora \
  --test  dataset/splits_carbot_1M/test.jsonl \
  --judge gpt-4o-mini \
  --out   ./outputs/judge_runs
```
Blind A/B with order flipping to reduce bias, JSON verdicts.

(Optional) Aggregate into Elo / Bradley–Terry scores.

---

## Reproducible Results (Example)
Test set of 590 samples, same hardware:
| Model |    EM | Slot Micro-F1 | Slot Macro-F1 | Schema Valid | Latency mean (s) | p95 (s) | VRAM peak (MiB) |
| ----- | ----: | ------------: | ------------: | -----------: | ---------------: | ------: | --------------: |
| Base  | 0.222 |         0.503 |         0.537 |        0.312 |             1.02 |    1.16 |           24927 |
| LoRA  | 0.222 |     **0.527** |     **0.569** |        0.312 |             1.51 |    1.72 |           24948 |

Interpretation: LoRA improves slot-level accuracy, but EM/schema stay flat due to naming/alias mismatches.
For deployment, prefer merged weights (lower latency).
Applying the normalization layer in evaluation usually boosts EM/schema.
