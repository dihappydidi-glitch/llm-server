#!/usr/bin/env python3
"""
Download any HuggingFace model (GGUF, EXL2, transformers, etc.)
For GTX 1660 6GB — recommends Qwen 2.5 7B (GGUF Q4_K_M)
Usage:  python download_model.py -m Qwen/Qwen2.5-7B-Instruct-GGUF
"""

import os
import sys
import argparse
from pathlib import Path

def download_model(model_name: str, target_dir: str, token: str = None):
    """Download a model from HuggingFace"""
    from huggingface_hub import snapshot_download

    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    final_path = target_path / model_name.split("/")[-1]

    print(f"Downloading: {model_name}")
    print(f"Target: {final_path}")
    print()

    kwargs = {
        "repo_id": model_name,
        "local_dir": str(final_path),
        "local_dir_use_symlinks": False,
        "resume_download": True,
        "force_download": False,
    }
    if token:
        kwargs["token"] = token

    result = snapshot_download(**kwargs)
    print(f"\nDownloaded to: {result}")

    # Estimate size
    total_size = sum(
        f.stat().st_size for f in final_path.rglob("*") if f.is_file()
    )
    print(f"Total size: {total_size / 1024**3:.2f} GiB")

    return result


def estimate_vram(model_name: str) -> str:
    """Estimate VRAM usage for a model"""
    name = model_name.lower()

    if "7b" in name or "8b" in name:
        if "3.2" in name and "3b" in name:
            return "~2.5 GiB (comfortable)"
        elif "1.5" in name or "1.6" in name:
            return "~2.5 GiB (comfortable)"
        return "~4.5-5.5 GiB (fits GTX 1660 6GB)"
    elif "3b" in name or "3.2" in name:
        return "~2-2.5 GiB (very comfortable)"
    elif "1b" in name or "1.5b" in name:
        return "~1-1.5 GiB (very comfortable)"
    else:
        return "unknown — check model card"


def main():
    parser = argparse.ArgumentParser(
        description="Download EXL2 model for ExLlamaV2"
    )
    parser.add_argument(
        "--model", "-m",
        default="turboderp/Qwen2.5-7B-Instruct-exl2",
        help="HuggingFace model ID (default: turboderp/Qwen2.5-7B-Instruct-exl2)"
    )
    parser.add_argument(
        "--dir", "-d",
        default="C:/Users/611marco/llm-server/models",
        help="Download directory"
    )
    parser.add_argument(
        "--token", "-t",
        default=None,
        help="HuggingFace token (not needed for most models)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="Show recommended models and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("=" * 60)
        print("  Recommended EXL2 models for GTX 1660 (6GB VRAM)")
        print("=" * 60)
        print()
        models = [
            ("turboderp/Qwen2.5-7B-Instruct-exl2",        "4.0 bpw", "~4.5 GiB", "✅ Best choice — great quality, fits well"),
            ("turboderp/Llama-3.1-8B-Instruct-exl2",       "4.0 bpw", "~5.5 GiB", "⚠️ Fits but tight — may need context limit"),
            ("bartowski/Mistral-7B-Instruct-v0.3-exl2",    "4.0 bpw", "~4.5 GiB", "✅ Good alternative"),
            ("turboderp/Llama-3.2-3B-Instruct-exl2",       "4.0 bpw", "~2.5 GiB", "✅ Fast, comfortable, less capable"),
            ("bartowski/Phi-3.5-mini-instruct-exl2",       "4.0 bpw", "~2.5 GiB", "✅ Fast, comfortable"),
        ]
        print(f"{'Model':<50} {'Quant':<8} {'VRAM':<10}  Notes")
        print("-" * 80)
        for m, q, v, note in models:
            print(f"{m:<50} {q:<8} {v:<10} {note}")
        print()
        print("Usage: python download_model.py -m turboderp/Qwen2.5-7B-Instruct-exl2")
        return

    print(f"Model: {args.model}")
    print(f"Estimated VRAM: {estimate_vram(args.model)}")
    print(f"Target: {args.dir}")
    print()

    download_model(args.model, args.dir, args.token)


if __name__ == "__main__":
    main()
