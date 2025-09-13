# Interactive demo with normalization + optional 4-bit
import os, time, re, json, copy, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

os.environ.setdefault("TORCHDYNAMO_DISABLE","1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir")
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit attempt")
    return ap.parse_args()

# Aliases
ACTION_ALIASES = {
    "car.system.set_lane_keeping_assist": "car.lks.set_main",
    "car.lane_keeping.set_main": "car.lks.set_main",
    "car.lks.set_level": "car.lks.set_assist_level",
    "car.audio.mute": "car.media.set_mute",
    "car.audio.set_volume": "car.media.set_volume",
    "car.media.volume.set": "car.media.set_volume",
    "car.windows.set_level": "car.window.set_level",
}
ARG_KEY_ALIASES = {"enable":"on","enabled":"on","volume":"level","lvl":"level"}
POSITION_ALIASES = {
    "driver":"front_left","driver_side":"front_left",
    "passenger":"front_right","passenger_side":"front_right",
    "left_rear":"rear_left","right_rear":"rear_right",
    "all_windows":"all","all window":"all","all_windows_full":"all",
}
ALLOWED_POSITIONS = {"front_left","front_right","rear_left","rear_right","all"}
LEVEL_BOUNDS = {
    "car.window.set_level": (1,3),
    "car.wipers.set": (1,3),
    "car.sunroof.set_level": (1,3),
    "car.seat.set_thermal": (1,3),
    "car.acc.set_headway_level": (1,3),
    "car.lks.set_assist_level": (1,3),
}

def normalize_action(action: dict):
    msgs=[]; n=copy.deepcopy(action)
    name = n.get("name")
    if isinstance(name,str) and name in ACTION_ALIASES:
        old=name; name=ACTION_ALIASES[name]; n["name"]=name; msgs.append(f"name: {old} -> {name}")
    args = n.get("args",{})
    if isinstance(args,dict):
        new={}
        for k,v in args.items():
            kk = ARG_KEY_ALIASES.get(k,k)
            if kk!=k: msgs.append(f"arg: {k} -> {kk}")
            new[kk]=v
        n["args"]=new; args=new
    pos = args.get("position")
    if isinstance(pos,str):
        p0=pos; p=pos.strip().lower().replace("-","_").replace(" ","_")
        p=POSITION_ALIASES.get(p,p)
        if p!=p0: msgs.append(f"position: {p0} -> {p}")
        n["args"]["position"]=p
        if p not in ALLOWED_POSITIONS: msgs.append(f"unknown position: {p}")
    if name in LEVEL_BOUNDS and "level" in args and isinstance(args["level"],int):
        lo,hi=LEVEL_BOUNDS[name]; lv=args["level"]
        if not(lo<=lv<=hi): msgs.append(f"level out of range for {name}: {lv} (allowed {lo}..{hi})")
    if "on" in args and not isinstance(args["on"],bool):
        msgs.append(f"'on' should be boolean")
    return n, msgs

SYSTEM = (
  "You are CarBot, an in-vehicle assistant.\n"
  "Return exactly two parts:\n"
  "1) <ACTION>{JSON}</ACTION>\n"
  "2) <SAY>short natural sentence</SAY>\n"
  "JSON schema: {\"name\":string, \"args\":{}}\n"
  "Allowed positions: front_left, front_right, rear_left, rear_right, all\n"
  "ACC/LKS levels are 1..3. Switches use {\"on\": true|false}.\n"
)
def build_prompt(user_text): return f"<SYSTEM>\n{SYSTEM}\n</SYSTEM>\n<USER>\n{user_text}\n</USER>\n<ASSISTANT>\n"
def extract_action_json(text):
    m=re.search(r"<ACTION>(\{.*?\})</ACTION>", text, flags=re.S); 
    if not m: return None
    try: return json.loads(m.group(1))
    except: return None
def pretty(d): 
    try: return json.dumps(d, ensure_ascii=False, indent=2)
    except: return str(d)

def main():
    args=parse_args()
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    print("[INFO] Loading tokenizer:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # try 4-bit
    m=None; quant=False
    if not args.no_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                         bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            m = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                     trust_remote_code=True, quantization_config=bnb_cfg)
            quant=True; print("[INFO] Loaded in 4-bit (nf4).")
        except Exception as e:
            print("[WARN] 4-bit failed, fallback to fp16/bf16:", e)
    if m is None:
        m = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
            torch_dtype=dtype, trust_remote_code=True)

    gc = GenerationConfig.from_model_config(m.config)
    gc.do_sample=False; gc.temperature=None; gc.top_p=None; gc.top_k=None
    m.generation_config=gc
    try: m.gradient_checkpointing_disable()
    except: pass
    if hasattr(m.config,"use_cache"): m.config.use_cache=True
    m.eval()

    print("\n[READY] merged model loaded.", "(4-bit)" if quant else f"({dtype})")
    print("Commands: /sample on|off, /temp <float>, /max <int>, /quit")
    sample=False; temperature=0.7; max_new_tokens=200

    while True:
        try: user=input("\nUSER > ").strip()
        except (EOFError,KeyboardInterrupt): print("\n[EXIT]"); break
        if not user: continue
        if user.lower() in ("/q","/quit","/exit"): print("[EXIT]"); break
        if user.lower().startswith("/sample"):
            parts=user.split(); 
            if len(parts)>=2 and parts[1].lower() in ("on","off"):
                sample=(parts[1].lower()=="on"); print("[CFG] sampling =", sample)
            else: print("[HELP] /sample on|off"); continue
            continue
        if user.lower().startswith("/temp"):
            parts=user.split()
            if len(parts)>=2:
                try: temperature=float(parts[1]); print("[CFG] temperature =", temperature)
                except: print("[HELP] /temp 0.7")
            else: print("[HELP] /temp 0.7")
            continue
        if user.lower().startswith("/max"):
            parts=user.split()
            if len(parts)>=2:
                try: max_new_tokens=int(parts[1]); print("[CFG] max_new_tokens =", max_new_tokens)
                except: print("[HELP] /max 200")
            else: print("[HELP] /max 200")
            continue

        prompt=build_prompt(user)
        enc=tok(prompt, return_tensors="pt").to(m.device)
        t0=time.time()
        with torch.inference_mode():
            out=m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=sample, **({"temperature":temperature} if sample else {}))
        el=time.time()-t0
        text=tok.decode(out[0], skip_special_tokens=True)
        assistant=text.split("<ASSISTANT>")[-1].strip()
        act=extract_action_json(assistant)

        print("\n--- OUTPUT (raw) ---------------------------")
        print(assistant)
        print("-------------------------------------------")

        if act is not None:
            norm, notes = normalize_action(act)
            if norm!=act:
                print("[ACTION JSON - RAW]");   print(pretty(act))
                print("[ACTION JSON - NORMALIZED]"); print(pretty(norm))
            else:
                print("[ACTION JSON]"); print(pretty(act))
            if notes:
                print("[Normalization Notes]"); [print(" -",n) for n in notes]
        else:
            print("[WARN] ACTION JSON not found or invalid.")

        print(f"[Latency] {el:.3f} s  | [Sampling] {sample}  | [MaxNew] {max_new_tokens}")

if __name__=="__main__":
    main()
