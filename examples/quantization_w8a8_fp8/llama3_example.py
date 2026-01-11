from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

DEFAULT_MODEL_PATH = "/home/scratch.mkilaru_coreai/Stockmark-2-100B-Instruct"
DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_PATH
DEFAULT_OUTPUT_ROOT = "/home/scratch.mkilaru_coreai"
DEFAULT_SCHEME = "FP8"

# Default calibration dataset (used for static FP8).
DEFAULT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize weights + activations to FP8 (compressed-tensors) using a local checkpoint.\n"
            "Defaults to the FP8 scheme (static, per-tensor).\n"
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "--scheme",
        choices=["FP8", "FP8_DYNAMIC"],
        default=DEFAULT_SCHEME,
        help="FP8 quantization preset to apply. Default is FP8 (static).",
    )

    # Calibration dataset options (required for --scheme FP8).
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
        help="Number of calibration samples to use (FP8 only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size (FP8 only).",
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)

    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")
    if not os.path.isdir(args.tokenizer_path):
        raise FileNotFoundError(f"--tokenizer-path not found: {args.tokenizer_path}")

    # Load model/tokenizer strictly from local paths (no HF download).
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

    # Configure FP8 quantization.
    # - FP8: static per-tensor (weights + activations)
    # - FP8_DYNAMIC: static weights + dynamic activations
    recipe = QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=["lm_head"])

    # Apply quantization.
    if args.scheme == "FP8":
        # Static FP8 requires calibration data.
        if args.dataset_path:
            ds = load_dataset("json", data_files=args.dataset_path, split="train")
            ds = ds.select(range(min(args.num_calibration_samples, len(ds))))  # type: ignore[arg-type]
        else:
            ds = load_dataset(
                args.dataset_id,
                split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
            )

        ds = ds.shuffle(seed=42)

        def preprocess(example):
            # Expect UltraChat-style {"messages": [...]} by default.
            if "messages" in example:
                text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
            elif "text" in example:
                text = example["text"]
            else:
                # Best-effort fallback.
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

        oneshot(
            model=model,
            tokenizer=tokenizer,
            dataset=ds,
            recipe=recipe,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
            batch_size=args.batch_size,
            trust_remote_code_model=True,
        )
    else:
        # FP8_DYNAMIC uses static FP8 weights + dynamic FP8 activations; no calibration required.
        oneshot(model=model, recipe=recipe)

    if not args.skip_generate:
        print("\n\n========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        sample = tokenizer("Hello my name is", return_tensors="pt")
        sample = {key: value.to(model.device) for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

    # Save to disk in compressed-tensors format.
    model_name = os.path.basename(os.path.normpath(args.model_path))
    suffix = "FP8" if args.scheme == "FP8" else "FP8-Dynamic"
    save_dir = args.save_dir or os.path.join(DEFAULT_OUTPUT_ROOT, f"{model_name}-{suffix}")
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved quantized model to: {save_dir}")


if __name__ == "__main__":
    main()
