import argparse, os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g., google/gemma-3-4b-it")
    ap.add_argument("--out", required=True, help="local dir")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=args.repo, local_dir=args.out, ignore_patterns=["*.msgpack","*.h5","*.onnx"])
    print("[OK] downloaded:", args.repo, "->", args.out)

if __name__ == "__main__":
    main()
