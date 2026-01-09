from __future__ import annotations

import argparse
import os

from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE:
# - The 405B checkpoint folder you have locally does not include tokenizer files.
# - Llama 3.1 uses a shared tokenizer across sizes, so we default to your local 8B tokenizer.
DEFAULT_MODEL_PATH = "/home/scratch.mkilaru_coreai/Llama-3.1-405B-Instruct"
DEFAULT_TOKENIZER_PATH = "/home/scratch.mkilaru_coreai/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_ROOT = "/home/scratch.mkilaru_coreai"

# Default calibration dataset (will download unless you override with --dataset-path).
DEFAULT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize Llama 3.1 405B using a local checkpoint.\n"
            "Supports:\n"
            "  - W4A16 (GPTQ INT4 weight-only)\n"
            "  - MXFP4 (experimental FP4 weight-only)\n"
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "--scheme",
        choices=["W4A16", "MXFP4"],
        default="W4A16",
        help="Quantization preset scheme to apply.",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional local dataset path (json/jsonl/parquet). If set, avoids downloading from HuggingFace Hub.",
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=128,
        help="Number of calibration samples to use (GPTQ only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Calibration batch size (GPTQ only). Increasing this can speed up calibration.",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")
    if not os.path.isdir(args.tokenizer_path):
        raise FileNotFoundError(f"--tokenizer-path not found: {args.tokenizer_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model_name = os.path.basename(os.path.normpath(args.model_path))

    if args.scheme == "W4A16":
        if args.dataset_path:
            ds = load_dataset("json", data_files=args.dataset_path, split="train")
            ds = ds.select(
                range(min(args.num_calibration_samples, len(ds)))  # type: ignore[arg-type]
            )
        else:
            ds = load_dataset(
                args.dataset_id,
                split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
            )

        ds = ds.shuffle(seed=42)

        def preprocess(example):
            if "messages" in example:
                text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            elif "text" in example:
                text = example["text"]
            else:
                text = str(example)
            return {"text": text}

        ds = ds.map(preprocess)

        def tokenize(sample):
            return tokenizer(
                sample["text"],
                padding=False,
                max_length=args.max_seq_length,
                truncation=True,
                add_special_tokens=False,
            )

        ds = ds.map(tokenize, remove_columns=ds.column_names)

        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])
        oneshot(
            model=model,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
            batch_size=args.batch_size,
        )
        default_suffix = "W4A16-G128"
    else:
        recipe = QuantizationModifier(
            targets="Linear", scheme="MXFP4", ignore=["lm_head"]
        )
        oneshot(model=model, recipe=recipe)
        default_suffix = "MXFP4"

    if not args.skip_generate:
        print("\n\n========== SAMPLE GENERATION ==============")
        dispatch_model(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample = {key: value.to(model.device) for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

    save_dir = args.save_dir or os.path.join(
        DEFAULT_OUTPUT_ROOT, f"{model_name}-{default_suffix}"
    )
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved quantized model to: {save_dir}")


if __name__ == "__main__":
    main()
