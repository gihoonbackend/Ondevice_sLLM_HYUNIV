import argparse, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out",  required=True)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto",
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
        trust_remote_code=True,
    )
    lora = PeftModel.from_pretrained(base, args.lora)
    merged = lora.merge_and_unload()

    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("[OK] merged to:", args.out)

if __name__ == "__main__":
    main()
