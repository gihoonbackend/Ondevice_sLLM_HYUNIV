import argparse, json, re
from pathlib import Path
from datasets import load_dataset
from collections import Counter

ACTION_RE = re.compile(r"<ACTION>(\{.*?\})</ACTION>", re.S)

def extract_action_json(txt):
    m = ACTION_RE.search(txt or "")
    if not m: return None
    try: return json.loads(m.group(1))
    except: return None

def add_intent(sample):
    intent = "unknown"
    for m in sample["messages"]:
        if m["role"] == "assistant":
            j = extract_action_json(m["content"])
            if j and isinstance(j, dict):
                intent = j.get("name", "unknown")
            break
    sample["intent"] = intent
    return sample

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("json", data_files=args.inp, split="train")
    labeled = raw.map(add_intent)
    labeled = labeled.class_encode_column("intent")

    # 80/10/10 stratified
    split_tmp = labeled.train_test_split(test_size=0.1, seed=42, stratify_by_column="intent")
    ds_rest, ds_test = split_tmp["train"], split_tmp["test"]
    split2 = ds_rest.train_test_split(test_size=0.111111, seed=42, stratify_by_column="intent")
    ds_train, ds_val = split2["train"], split2["test"]

    ds_train.to_json(str(out_dir/"train.jsonl"))
    ds_val.to_json(str(out_dir/"val.jsonl"))
    ds_test.to_json(str(out_dir/"test.jsonl"))

    def dist(ds):
        c = Counter(ds["intent"])
        return dict(sorted(c.items(), key=lambda x: x[0]))

    print("sizes:", len(ds_train), len(ds_val), len(ds_test))
    print("saved to:", out_dir)

if __name__ == "__main__":
    main()
