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
- [CarBot Inference Examples]
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
├─ dataset/
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

## Evaluation Metrics — Rationale
1) Correctness & Format

EM (Exact Match)
Checks whether the <ACTION>{…}</ACTION> JSON is exactly identical to the gold reference (character-for-character).
Why: In vehicle control, a single parameter mismatch (e.g., level 2 vs 3) can execute the wrong action—“close enough” is not acceptable.

Slot Micro/Macro-F1
Captures partial correctness beyond all-or-nothing EM.

Micro-F1: aggregates over all slots (key–value pairs) → overall quality.

Macro-F1: averages across samples/intents → sensitive to rare/hard cases.
Why: Training often improves slot accuracy first; EM alone can miss this progress.

Schema Valid Rate
Share of outputs that conform to the allowed schema/ranges (levels 1..3, booleans, valid positions, etc.).
Why: Commands go straight to ECU/gateway; invalid format ⇒ immediate failure. Executability is quality.

2) Real-time & On-device Constraints

Latency (mean / p95)
Mean = typical speed; p95 = tail latency (perceived “stutter”).
Why: Infotainment/ADAS control needs snappy UX; high p95 hurts user perception.

VRAM Peak
Peak GPU memory usage for deployability on constrained devices.
Why: Quantifies trade-offs from 4-bit/8-bit quantization or LoRA/merge.

3) Task-Specific Choices

Why not BLEU/ROUGE?
Goal is correct control JSON, not fluent text; similarity metrics poorly reflect control correctness.

Why no separate Intent Accuracy?
Largely reflected by EM/Slot metrics and a clear action schema.
(Easy to add later—e.g., “action-name match rate.”)

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

## NO_HINT_PROMPT
```
  "You are CarBot, an in-vehicle assistant.\n"
  "Return exactly two parts in this order:\n"
  "1) <ACTION>{JSON}</ACTION>\n"
  "2) <SAY>short natural sentence</SAY>\n"
  "JSON schema: {\"name\": string, \"args\": {object}}\n"
  "Constraints:\n"
  "- STRICT JSON (no natural language in values). No extra keys.\n"
  "- Positions must be valid; booleans for switches; integer levels.\n"
```

| Model |    EM | Slot Micro-F1 | Slot Macro-F1 | Schema Valid | Latency mean (s) |   p95 (s) | VRAM peak (MiB) |
| ----- | ----: | ------------: | ------------: | -----------: | ---------------: | --------: | --------------: |
| Base  | 0.000 |         0.349 |         0.355 |        0.000 |            1.026 |     1.821 |            8256 |
| LoRA  | 0.000 |     **0.375** |     **0.382** |        0.000 |        **0.967** | **1.820** |            8255 |

## taxonomy_hint_PROMPT
```
  "You are CarBot, an in-vehicle assistant.\n"
  "Return exactly two parts in this order:\n"
  "1) <ACTION>{JSON}</ACTION>\n"
  "2) <SAY>short natural sentence</SAY>\n"
  "JSON schema: {\"name\": string, \"args\": {object}}\n"
  "Constraints:\n"
  "- STRICT JSON (no natural language in values). No extra keys.\n"
  "- Positions: front_left, front_right, rear_left, rear_right, all\n"
  "- ACC/LKS levels are 1..3. Switches use {\"on\": true|false}.\n"
  "Action taxonomy (allowed names & args):\n"
  "- car.window.set_level: {\"position\": front_left|front_right|rear_left|rear_right|all, \"level\": 1..3}\n"
  "- car.window.switch: {\"position\": front_left|front_right|rear_left|rear_right|all, \"on\": true|false}\n"
  "- car.media.set_volume: {\"level\": 0..10}\n"
  "- car.media.set_mute: {\"on\": true|false}\n"
  "- car.media.command: {\"name\": play|pause|next|previous}\n"
  "- car.seat.set_thermal: {\"position\": driver|passenger|rear_left|rear_right, \"level\": 1..3}\n"
  "- car.steering_wheel.set_heater: {\"on\": true|false}\n"
  "- car.sunroof.set_level: {\"level\": 1..3}\n"
  "- car.lights.set: {\"on\": true|false}\n"
  "- car.wipers.set: {\"level\": 1..3}\n"
  "- car.acc.set_main: {\"on\": true|false}\n"
  "- car.acc.set_headway_level: {\"level\": 1..3}\n"
  "- car.lks.set_main: {\"on\": true|false}\n"
  "- car.lks.set_assist_level: {\"level\": 1..3}\n"
  "- ask.clarify: {\"reason\": string}\n"
  "Examples:\n"
  "<ACTION>{\"name\":\"car.acc.set_main\",\"args\":{\"on\":true}}</ACTION>\n"
  "<SAY>ACC is on.</SAY>\n"
  "<ACTION>{\"name\":\"car.acc.set_headway_level\",\"args\":{\"level\":3}}</ACTION>\n"
  "<SAY>Following distance set to level 3.</SAY>\n"
  "<ACTION>{\"name\":\"car.window.set_level\",\"args\":{\"position\":\"rear_right\",\"level\":2}}</ACTION>\n"
  "<SAY>Opening the right rear window to level 2.</SAY>\n"
```
| Model |        EM | Slot Micro-F1 | Slot Macro-F1 | Schema Valid | Latency mean (s) | p95 (s) | VRAM peak (MiB) |
| ----- | --------: | ------------: | ------------: | -----------: | ---------------: | ------: | --------------: |
| Base  |     0.500 |         0.552 |         0.609 |        0.890 |            1.048 |   1.189 |            8350 |
| LoRA  | **0.510** |     **0.560** |     **0.619** |    **0.890** |            1.050 |   1.188 |            8350 |


Interpretation: LoRA improves slot-level accuracy, but EM/schema stay flat due to naming/alias mismatches.
For deployment, prefer merged weights (lower latency).
Applying the normalization layer in evaluation usually boosts EM/schema.


---

## CarBot Inference Examples
```
1) Make volume 6

User: Make volume 6

Action JSON:

{
  "name": "car.media.set_volume",
  "args": { "level": 6 }
}


Say: Volume set to 6.

Latency: 1.359 s

2) Open the driver window

User: open the driver window

Action JSON:

{
  "name": "car.window.set_level",
  "args": { "position": "driver", "level": 2 }
}


Say: Opening the driver window to level 2.

Latency: 1.121 s

3) Turn on LKS

User: turn on lks

Action JSON:

{
  "name": "car.lks.set_main",
  "args": { "on": true }
}


Say: LKS is now on.

Latency: 0.998 s

4) Turn on ACC 

User: turn on acc 

Action JSON:

{
  "name": "car.acc.set_main",
  "args": { "on": true }
}


Say: ACC is on.

Latency: 0.958 s

5) Turn on ACC level 2

User: turn on acc level 2

Action JSON:

{
  "name": "car.acc.set_headway_level",
  "args": { "level": 2 }
}


Say: ACC headway level set to 2.

Latency: 1.078 s

6) Turn off light

User: turn off light

Action JSON:

{
  "name": "car.lights.set",
  "args": { "on": false }
}


Say: Lights are off.

Latency: 0.913 s
```
