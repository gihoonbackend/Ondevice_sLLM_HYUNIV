# Compatible with: accelerate 1.2.1, transformers 4.54.1, trl 0.17.0, peft 0.17.0
import argparse, json, os, torch, re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/torchinductor"))

ACTION_RE = re.compile(r"<ACTION>(\{.*?\})</ACTION>", re.S)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train = load_dataset("json", data_files=args.train, split="train")
    val   = load_dataset("json", data_files=args.val,   split="train")

    # Flatten to {"text": "..."} with system wrap
    SYSTEM = (
      "You are CarBot, an in-vehicle assistant.\n"
      "Return exactly two parts:\n"
      "1) <ACTION>{JSON}</ACTION>\n"
      "2) <SAY>short natural sentence</SAY>\n"
      "JSON schema: {\"name\":string,\"args\":{}}\n"
      "Allowed positions: front_left, front_right, rear_left, rear_right, all\n"
      "ACC/LKS levels: 1..3. Switches: {\"on\": true|false}.\n"
    )
    def to_text(sample):
      user = next(m["content"] for m in sample["messages"] if m["role"]=="user")
      asst = next(m["content"] for m in sample["messages"] if m["role"]=="assistant")
      return {"text": f"<SYSTEM>\n{SYSTEM}\n</SYSTEM>\n<USER>\n{user}\n</USER>\n<ASSISTANT>\n{asst}"}
    train_f = train.map(to_text, remove_columns=train.column_names)
    val_f   = val.map(to_text,   remove_columns=val.column_names)

    # LoRA
    peft_cfg = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )

    # SFT config
    sft_cfg = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_grad_norm=1.0,
        logging_steps=50,
        save_steps=0,                 # we’ll save at the end
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_len,
        packing=True,
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<ACTION>", tokenizer=tokenizer
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto",
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
        trust_remote_code=True,
    )

    trainer = SFTTrainer(
        model=base_model,
        args=sft_cfg,
        train_dataset=train_f,
        eval_dataset=val_f,
        peft_config=peft_cfg,
        processing_class=tokenizer,
        data_collator=collator,
    )

    # Workaround for accelerate < keep_torch_compile kwarg
    try:
        import inspect
        orig_unwrap = trainer.accelerator.unwrap_model
        if "keep_torch_compile" not in inspect.signature(orig_unwrap).parameters:
            def _unwrap_no_kwargs(model, *a, **kw): return orig_unwrap(model)
            trainer.accelerator.unwrap_model = _unwrap_no_kwargs
    except Exception:
        pass

    res = trainer.train()
    trainer.save_model()
    trainer.save_state()
    print("[DONE] train metrics:", res.metrics)

if __name__ == "__main__":
    main()
