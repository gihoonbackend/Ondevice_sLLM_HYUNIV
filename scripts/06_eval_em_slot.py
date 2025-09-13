# Quant metrics: EM / Slot-F1 / Schema / Latency / VRAM
import argparse, json, time, re, numpy as np, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

ACTION_RE = re.compile(r"<ACTION>(\{.*?\})</ACTION>", re.S)

ALLOWED_POS = {"front_left","front_right","rear_left","rear_right","all"}
ACTION_SPECS = {
    "car.window.set_level": {"args": {"position": ALLOWED_POS, "level": ("int",1,3)}},
    "car.window.switch":    {"args": {"position": ALLOWED_POS, "on": ("bool",)}},
    "car.media.set_volume": {"args": {"level": ("int",0,10)}},   # adjust as needed
    "car.media.set_mute":   {"args": {"on": ("bool",)}},
    "car.media.command":    {"args": {"name": ("enum", {"play","pause","next","previous"})}},
    "car.seat.set_thermal": {"args": {"position": {"driver","passenger","rear_left","rear_right"}, "level": ("int",1,3)}},
    "car.steering_wheel.set_heater": {"args": {"on": ("bool",)}},
    "car.sunroof.set_level":{"args": {"level": ("int",1,3)}},
    "car.lights.set":       {"args": {"on": ("bool",)}},
    "car.wipers.set":       {"args": {"level": ("int",1,3)}},
    "car.acc.set_main":            {"args": {"on": ("bool",)}},
    "car.acc.set_headway_level":   {"args": {"level": ("int",1,3)}},
    "car.lks.set_main":            {"args": {"on": ("bool",)}},
    "car.lks.set_assist_level":    {"args": {"level": ("int",1,3)}},
    "ask.clarify": {"args": {"reason": ("str",)}},
}

def extract_action_json(text):
    m=ACTION_RE.search(text or ""); 
    if not m: return None
    try: return json.loads(m.group(1))
    except: return None

def get_user_msg(sample): return next((m["content"] for m in sample["messages"] if m["role"]=="user"), "")
def get_target(sample):  return extract_action_json(next((m["content"] for m in sample["messages"] if m["role"]=="assistant"), ""))

def validate_action(a:dict):
    errs=[]
    if not isinstance(a,dict) or "name" not in a or "args" not in a: return False,["missing name/args"]
    name=a["name"]; args=a["args"]; spec=ACTION_SPECS.get(name)
    if not spec: return False,[f"unknown action: {name}"]
    allowed=set(spec["args"].keys())
    if set(args.keys())-allowed: errs.append(f"extra keys: {list(set(args.keys())-allowed)}")
    for k,rule in spec["args"].items():
        if k not in args: errs.append(f"missing arg: {k}"); continue
        v=args[k]
        if isinstance(rule,set):
            if v not in rule: errs.append(f"{k} not in {sorted(rule)}: {v}")
        elif isinstance(rule,tuple):
            t=rule[0]
            if t=="bool":
                if not isinstance(v,bool): errs.append(f"{k} not bool: {v}")
            elif t=="int":
                lo,hi=rule[1],rule[2]
                if not isinstance(v,int): errs.append(f"{k} not int: {v}")
                elif not(lo<=v<=hi): errs.append(f"{k} out of range [{lo},{hi}]: {v}")
            elif t=="enum":
                if v not in rule[1]: errs.append(f"{k} not in {sorted(rule[1])}: {v}")
            elif t=="str":
                if not isinstance(v,str): errs.append(f"{k} not str")
    return (len(errs)==0), errs

def compare_slots(pred, tgt):
    if pred is None and tgt is None: return (0,0,0),(1,1,1)
    if pred is None: return (0,0,len(tgt) if tgt else 0),(0,0,0)
    if tgt  is None: return (0,len(pred) if pred else 0,0),(0,0,0)
    p_args=pred.get("args",pred); t_args=tgt.get("args",tgt)
    p=set((k,json.dumps(v,sort_keys=True)) for k,v in p_args.items())
    t=set((k,json.dumps(v,sort_keys=True)) for k,v in t_args.items())
    tp=len(p&t); fp=len(p-t); fn=len(t-p)
    prec=tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec =tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1  =2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return (tp,fp,fn),(prec,rec,f1)

def prep_model_for_infer(model):
    if hasattr(model,"_orig_mod"): model=model._orig_mod
    try: model.gradient_checkpointing_disable()
    except: pass
    if hasattr(model.config,"use_cache"): model.config.use_cache=True
    model.eval()
    gc=GenerationConfig.from_model_config(model.config)
    gc.do_sample=False; gc.temperature=None; gc.top_p=None; gc.top_k=None
    model.generation_config=gc
    return model

def load_base(mid, dtype):
    m=AutoModelForCausalLM.from_pretrained(mid, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
    t=AutoTokenizer.from_pretrained(mid, use_fast=True, trust_remote_code=True)
    if t.pad_token is None: t.pad_token=t.eos_token
    return prep_model_for_infer(m), t

def load_lora(base_id, lora_dir, dtype):
    base=AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", torch_dtype=dtype, trust_remote_code=True)
    m=PeftModel.from_pretrained(base, lora_dir)
    t=AutoTokenizer.from_pretrained(base_id, use_fast=True, trust_remote_code=True)
    if t.pad_token is None: t.pad_token=t.eos_token
    return prep_model_for_infer(m), t

def gen_once(model, tok, prompt, max_new=200):
    dev=next(model.parameters()).device
    enc=tok(prompt, return_tensors="pt").to(dev)
    t0=time.time()
    out=model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    lat=time.time()-t0
    txt=tok.decode(out[0], skip_special_tokens=True)
    asst=txt.split("<ASSISTANT>")[-1].strip()
    return asst, lat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=200)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    ds = load_dataset("json", data_files=args.test, split="train")

    SYSTEM=(
      "You are CarBot, an in-vehicle assistant.\n"
      "Return exactly two parts:\n"
      "1) <ACTION>{JSON}</ACTION>\n"
      "2) <SAY>short natural sentence</SAY>\n"
      "JSON schema: {\"name\": string, \"args\": {}}\n"
      "Allowed positions: front_left, front_right, rear_left, rear_right, all\n"
      "ACC/LKS levels are 1..3. Switches use {\"on\": true|false}.\n"
    )
    def prompt_of(user): return f"<SYSTEM>\n{SYSTEM}\n</SYSTEM>\n<USER>\n{user}\n</USER>\n<ASSISTANT>\n"

    # data
    prompts=[]; gts=[]
    for s in ds:
        user=get_user_msg(s); gt=get_target(s)
        prompts.append(user); gts.append(gt)

    # base
    base_m, base_t = load_base(args.base, dtype)
    # vram peak
    if torch.cuda.is_available():
        device=next(base_m.parameters()).device
        torch.cuda.reset_peak_memory_stats(device)

    ok=0; valid=0; lat=[]
    micro_tp=micro_fp=micro_fn=0; macro_f=[]
    for i,u in enumerate(prompts):
        out, l = gen_once(base_m, base_t, prompt_of(u), args.max_new)
        pred=extract_action_json(out)
        v,_=validate_action(pred) if pred is not None else (False,["none"])
        valid+=int(v)
        gt=gts[i]
        if pred==gt: ok+=1
        (tp,fp,fn),(pr,re,f1)=compare_slots(pred, gt)
        micro_tp+=tp; micro_fp+=fp; micro_fn+=fn; macro_f.append(f1); lat.append(l)
    em_base=ok/len(prompts)
    def micro_f1(tp,fp,fn):
        p=tp/(tp+fp) if (tp+fp)>0 else 0.0
        r=tp/(tp+fn) if (tp+fn)>0 else 0.0
        return 2*p*r/(p+r+1e-12)
    res_base={
        "n":len(prompts),
        "schema_valid_rate": valid/len(prompts),
        "latency":{"mean":float(np.mean(lat)), "p95":float(np.percentile(lat,95))},
        "vram_peak_MiB": (torch.cuda.max_memory_allocated(device)/(1024**2)) if torch.cuda.is_available() else None,
        "EM": em_base,
        "slot_micro_f1": micro_f1(micro_tp,micro_fp,micro_fn),
        "slot_macro_f1": float(np.mean(macro_f)),
    }

    # lora
    lora_m, lora_t = load_lora(args.base, args.lora, dtype)
    if torch.cuda.is_available():
        device=next(lora_m.parameters()).device
        torch.cuda.reset_peak_memory_stats(device)

    ok=0; valid=0; lat=[]
    micro_tp=micro_fp=micro_fn=0; macro_f=[]
    for i,u in enumerate(prompts):
        out, l = gen_once(lora_m, lora_t, prompt_of(u), args.max_new)
        pred=extract_action_json(out)
        v,_=validate_action(pred) if pred is not None else (False,["none"])
        valid+=int(v)
        gt=gts[i]
        if pred==gt: ok+=1
        (tp,fp,fn),(pr,re,f1)=compare_slots(pred, gt)
        micro_tp+=tp; micro_fp+=fp; micro_fn+=fn; macro_f.append(f1); lat.append(l)
    em_lora=ok/len(prompts)
    res_lora={
        "n":len(prompts),
        "schema_valid_rate": valid/len(prompts),
        "latency":{"mean":float(np.mean(lat)), "p95":float(np.percentile(lat,95))},
        "vram_peak_MiB": (torch.cuda.max_memory_allocated(device)/(1024**2)) if torch.cuda.is_available() else None,
        "EM": em_lora,
        "slot_micro_f1": micro_f1(micro_tp,micro_fp,micro_fn),
        "slot_macro_f1": float(np.mean(macro_f)),
    }

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out)/"metrics_summary.json","w") as f:
        json.dump({"base":res_base, "lora":res_lora}, f, indent=2)
    print(json.dumps({"base":res_base,"lora":res_lora}, indent=2))
    print("[SAVED] metrics_summary.json ->", args.out)

if __name__ == "__main__":
    main()
