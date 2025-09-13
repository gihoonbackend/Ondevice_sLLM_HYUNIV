# LLM-as-a-Judge: A/B blind with order flip, optional ref-aware judging
import os, argparse, json, re, random, time, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from openai import OpenAI

os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))

def extract_action_json(text):
    m=re.search(r"<ACTION>(\{.*?\})</ACTION>", text or "", flags=re.S)
    if not m: return None
    try: return json.loads(m.group(1))
    except: return None

SYSTEM_TASK = (
  "You are CarBot, an in-vehicle assistant.\n"
  "Return exactly two parts:\n"
  "1) <ACTION>{JSON}</ACTION>\n"
  "2) <SAY>short natural sentence</SAY>\n"
  "JSON schema: {'name':string,'args':{}}\n"
  "Allowed positions: front_left, front_right, rear_left, rear_right, all\n"
  "ACC/LKS levels 1..3; switches use {'on': true|false}\n"
)
def build_prompt(user): return f"<SYSTEM>\n{SYSTEM_TASK}\n</SYSTEM>\n<USER>\n{user}\n</USER>\n<ASSISTANT>\n"

SYSTEM_FOR_JUDGE = (
  "You are an expert evaluator for in-vehicle assistants. Compare two answers (A,B) and return a strict JSON verdict."
)
def build_judge_prompt(user_text, cand_A, cand_B, gt_json=None, ref_aware=True):
    rubric = (
      "- Format: <ACTION>{JSON}</ACTION> present, valid JSON, allowed keys only.\n"
      "- Correctness: JSON matches intended action; values in allowed ranges.\n"
      "- Utility: helpful, concise.\n"
      "- Safety: avoids unsafe/irrelevant operations.\n"
    )
    ref = f"Ground truth JSON:\n{json.dumps(gt_json)}\n" if (ref_aware and gt_json is not None) else "No ground truth provided.\n"
    return f"""You will see a user request and two candidate assistant replies (A and B).
{ref}
Evaluation rubric:
{rubric}
Return ONLY a JSON object with this schema:
{{
  "winner": "A" | "B" | "tie",
  "reasons": "short explanation (max 3 bullet points)",
  "scores": {{"format": 0-1, "correctness": 0-1, "utility": 0-1, "safety": 0-1}}
}}

User:
{user_text}

Candidate A:
{cand_A}

Candidate B:
{cand_B}
"""

def load_and_prep(mid, dtype):
    m=AutoModelForCausalLM.from_pretrained(mid, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
    t=AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=True)
    if t.pad_token is None: t.pad_token=t.eos_token
    gc=GenerationConfig.from_model_config(m.config)
    gc.do_sample=False; gc.temperature=None; gc.top_p=None; gc.top_k=None
    m.generation_config=gc
    if hasattr(m,"_orig_mod"): m=m._orig_mod
    try: m.gradient_checkpointing_disable()
    except: pass
    if hasattr(m.config,"use_cache"): m.config.use_cache=True
    m.eval()
    return m,t

def attach_lora(base_id, lora_dir, dtype):
    base=AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
    m=PeftModel.from_pretrained(base, lora_dir)
    t=AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
    if t.pad_token is None: t.pad_token=t.eos_token
    gc=GenerationConfig.from_model_config(m.config)
    gc.do_sample=False; gc.temperature=None; gc.top_p=None; gc.top_k=None
    m.generation_config=gc
    m.eval()
    return m,t

def generate(model, tok, user, max_new=200):
    enc=tok(build_prompt(user), return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        out=model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    txt=tok.decode(out[0], skip_special_tokens=True)
    return txt.split("<ASSISTANT>")[-1].strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=200)
    args=ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)

    ds=load_dataset("json", data_files=args.test, split="train")
    prompts=[]; gts=[]
    for s in ds:
        user = next(m["content"] for m in s["messages"] if m["role"]=="user")
        gt   = extract_action_json(next(m["content"] for m in s["messages"] if m["role"]=="assistant"))
        prompts.append(user); gts.append(gt)

    base_m, base_t = load_and_prep(args.base, dtype)
    lora_m, lora_t = attach_lora(args.base, args.lora, dtype)

    wins={"A":0,"B":0,"tie":0}; rows=[]
    for i, u in enumerate(prompts):
        # produce two answers
        a = generate(base_m, base_t, u, args.max_new)
        b = generate(lora_m, lora_t, u, args.max_new)

        # randomize order & flip twice
        for flip in (0,1):
            A, B = (a,b) if flip==0 else (b,a)
            metaA, metaB = ( "base","lora") if flip==0 else ("lora","base")
            prompt_j = build_judge_prompt(u, A, B, gt_json=gts[i], ref_aware=True)
            resp = client.chat.completions.create(
                model=args.judge, temperature=0.0,
                messages=[{"role":"system","content":"You are a strict JSON-only judge."},
                          {"role":"user","content":prompt_j}]
            )
            text = resp.choices[0].message.content
            try:
                verdict = json.loads(text)
            except Exception:
                m=re.search(r"\{[\s\S]*\}$", text.strip())
                verdict=json.loads(m.group(0)) if m else {"winner":"tie","reasons":"parse_fail","scores":{}}
            w = verdict.get("winner","tie")
            wins[w]=wins.get(w,0)+1
            rows.append({"i":i,"user":u,"A_model":metaA,"B_model":metaB,"A":A,"B":B,"verdict":verdict})

    ts=int(time.time())
    with open(Path(args.out)/f"ab_judge_base_lora_{ts}.jsonl","w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with open(Path(args.out)/f"ab_judge_wins_{ts}.json","w") as f:
        json.dump(wins, f, indent=2)
    print("wins:", wins)
    print("[SAVED] judge results ->", args.out)

if __name__=="__main__":
    main()
