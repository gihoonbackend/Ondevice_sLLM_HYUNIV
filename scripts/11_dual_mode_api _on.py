#!/usr/bin/env python3
import os, time, re, json, copy, argparse, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from openai import OpenAI

os.environ.setdefault("TORCHDYNAMO_DISABLE", "0")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",
                     os.path.expanduser("~/.cache/torchinductor"))

# =====================================================
# 모델 경로 (머지된 모델만 사용)
# =====================================================
DEFAULT_MERGED_MODEL_PATH = "model_path"

# OpenAI 모델 이름
OPENAI_MODEL_NAME = "gpt-4o-mini"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MERGED_MODEL_PATH,
                    help="path to merged LoRA model")
    return ap.parse_args()


# =====================================================
# 차량 제어용 Aliases
# =====================================================
ACTION_ALIASES = {
    "car.system.set_lane_keeping_assist": "car.lks.set_main",
    "car.lane_keeping.set_main": "car.lks.set_main",
    "car.lks.set_level": "car.lks.set_assist_level",
    "car.audio.mute": "car.media.set_mute",
    "car.audio.set_volume": "car.media.set_volume",
    "car.media.volume.set": "car.media.set_volume",
    "car.windows.set_level": "car.window.set_level",
}

ARG_KEY_ALIASES = {"enable": "on", "enabled": "on",
                   "volume": "level", "lvl": "level"}

POSITION_ALIASES = {
    "driver": "front_left", "driver_side": "front_left",
    "passenger": "front_right", "passenger_side": "front_right",
    "left_rear": "rear_left", "right_rear": "rear_right",
    "all_windows": "all", "all window": "all",
    "all_windows_full": "all",
}

ALLOWED_POSITIONS = {
    "front_left", "front_right", "rear_left", "rear_right", "all"}

LEVEL_BOUNDS = {
    "car.window.set_level": (1, 3),
    "car.wipers.set": (1, 3),
    "car.sunroof.set_level": (1, 3),
    "car.seat.set_thermal": (1, 3),
    "car.acc.set_headway_level": (1, 3),
    "car.lks.set_assist_level": (1, 3),
}


def normalize_action(action: dict):
    msgs = []
    n = copy.deepcopy(action)
    name = n.get("name")
    if isinstance(name, str) and name in ACTION_ALIASES:
        old = name
        name = ACTION_ALIASES[name]
        n["name"] = name
        msgs.append(f"name: {old} -> {name}")

    args = n.get("args", {})
    if isinstance(args, dict):
        new = {}
        for k, v in args.items():
            kk = ARG_KEY_ALIASES.get(k, k)
            if kk != k:
                msgs.append(f"arg: {k} -> {kk}")
            new[kk] = v
        n["args"] = new
        args = new

    pos = args.get("position")
    if isinstance(pos, str):
        p0 = pos
        p = pos.strip().lower().replace("-", "_").replace(" ", "_")
        p = POSITION_ALIASES.get(p, p)
        if p != p0:
            msgs.append(f"position: {p0} -> {p}")
        n["args"]["position"] = p
        if p not in ALLOWED_POSITIONS:
            msgs.append(f"unknown position: {p}")

    if name in LEVEL_BOUNDS and "level" in args and isinstance(args["level"], int):
        lo, hi = LEVEL_BOUNDS[name]
        lv = args["level"]
        if not (lo <= lv <= hi):
            msgs.append(f"level out of range for {name}: {lv} (allowed {lo}..{hi})")

    if "on" in args and not isinstance(args["on"], bool):
        msgs.append(f"'on' should be boolean")

    return n, msgs


def extract_action_json(text: str):
    m = re.search(r"<ACTION>(\{.*?\})</ACTION>", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def pretty(d):
    try:
        return json.dumps(d, ensure_ascii=False, indent=2)
    except Exception:
        return str(d)


# =====================================================
# SYSTEM 프롬프트
# =====================================================
SYSTEM_CARBOT = (
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
    "- car.window.set_level: {\"target\": front_left|front_right|rear_left|rear_right|all, \"level\": 1..3}\n"
    "- car.window.switch: {\"target\": front_left|front_right|rear_left|rear_right|all, \"on\": true|false}\n"
    "- car.media.set_volume: {\"level\": 0..15}\n"
    "- car.media.set_mute: {\"on\": true|false}\n"
    "- car.media.command: {\"cmd\": play|pause|next|previous|stop}\n"
    "- car.seat.set_thermal: {\"seat\": driver|passenger|rear_left|rear_right, \"feature\": vent|heat, \"level\": 1..3}\n"
    "- car.steering_wheel.set_heater: {\"on\": true|false}\n"
    "- car.sunroof.set_level: {\"level\": 1..3}\n"
    "- car.lights.set: {\"mode\": off|auto|on, \"high_beam\": true|false}\n"
    "- car.wipers.set: {\"mode\": off|auto|on, \"level\": 1..3}\n"
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
    "<ACTION>{\"name\":\"car.window.set_level\",\"args\":{\"target\":\"rear_right\",\"level\":2}}</ACTION>\n"
)

SYSTEM_CHAT = (
    "You are a friendly in-car assistant. "
    "Reply naturally and conversationally. "
    "Do NOT output any JSON or car control commands. "
    "Keep answers short, helpful, and safe."
)


def build_carbot_prompt(user_text: str) -> str:
    return f"<SYSTEM>\n{SYSTEM_CARBOT}\n</SYSTEM>\n<USER>\n{user_text}\n</USER>\n<ASSISTANT>\n"


def build_chat_prompt(user_text: str) -> str:
    return f"<SYSTEM>\n{SYSTEM_CHAT}\n</SYSTEM>\n<USER>\n{user_text}\n</USER>\n<ASSISTANT>\n"


# =====================================================
# Domain Classifier
# =====================================================
CARBOT_KEYWORDS = [
    "acc", "adaptive cruise", "cruise", "headway",
    "lane keep", "lks", "lane keeping",
    "volume", "media", "mute", "radio",
    "window", "sunroof",
    "seat", "heater", "vent",
    "light", "headlight",
    "wiper",
]


def is_car_command(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CARBOT_KEYWORDS)


# =====================================================
# OpenAI Chat
# =====================================================
def openai_chat(user_text: str):
    api_key = "open-api-key"

    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_CHAT},
                {"role": "user", "content": user_text},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[OpenAI ERROR] {e}"


# =====================================================
# MAIN
# =====================================================
def main():
    args = parse_args()

    dtype = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ) else torch.float16

    print("[INFO] Loading merged CarBot model:", args.model)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    merge_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
    )
    gc = GenerationConfig.from_model_config(merge_model.config)
    gc.do_sample = False
    merge_model.generation_config = gc
    merge_model.eval()

    print("\n[READY] Single merged model + OpenAI Chat loaded.")

    while True:
        try:
            user = input("\nUSER > ").strip()
        except:
            print("\n[EXIT]")
            break

        if not user:
            continue
        if user.lower() in ("/q", "/quit", "/exit"):
            break

        control_mode = is_car_command(user)

        # -------------------------
        # Control Mode → merged local model
        # -------------------------
        if control_mode:
            print("[MODE] Car Control (Merged Model)")
            prompt = build_carbot_prompt(user)
            enc = tok(prompt, return_tensors="pt").to(merge_model.device)

            with torch.inference_mode():
                out = merge_model.generate(
                    **enc,
                    max_new_tokens=200,
                    do_sample=False
                )

            text = tok.decode(out[0], skip_special_tokens=True)
            assistant = text.split("<ASSISTANT>")[-1].strip()

            print("\n--- OUTPUT ---------------------")
            print(assistant)
            print("--------------------------------")

            act = extract_action_json(assistant)
            if act:
                norm, notes = normalize_action(act)
                print("[ACTION JSON]")
                print(pretty(norm))
                if notes:
                    print("[Normalization Notes]")
                    for n in notes:
                        print(" -", n)
            else:
                print("[WARN] ACTION JSON not found")

        # -------------------------
        # Chat Mode → OpenAI API
        # -------------------------
        else:
            print("[MODE] Chat (OpenAI API)")
            reply = openai_chat(user)
            print("\n--- CHAT OUTPUT ----------------")
            print(reply)
            print("--------------------------------")


if __name__ == "__main__":
    main()
