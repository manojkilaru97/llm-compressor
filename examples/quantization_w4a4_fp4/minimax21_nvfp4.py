from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation


DEFAULT_MODEL_PATH = "/home/scratch.mkilaru_coreai/MiniMax-M2.1"
DEFAULT_OUTPUT_DIR = "/home/scratch.mkilaru_coreai/MiniMax-M2.1-NVFP4"

# A small chat dataset for calibrating activation global scales.
DEFAULT_DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"


def _apply_chat_template(tokenizer, messages) -> str:
    # Prefer tokenizer chat template if present; fall back to a simple join.
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass
    return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize MiniMax-2.1 to NVFP4 (Blackwell FP4, W4A4) using llm-compressor oneshot. "
            "This calibrates activation global scales from a small dataset and saves in "
            "compressed-tensors format for vLLM."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--save-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--num-calibration-samples", type=int, default=20)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip a small post-quantization generation sanity check.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), load model/tokenizer strictly from local checkpoint files.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (default), allow custom model code for MiniMax checkpoints.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    # Load dataset and preprocess to text.
    ds = load_dataset(
        args.dataset_id,
        split=f"{args.dataset_split}[:{args.num_calibration_samples}]",
    ).shuffle(seed=42)

    def preprocess(example):
        messages = example.get("messages")
        if messages is None:
            # Fallback: best-effort; some datasets may have different schema.
            text = example.get("text") or example.get("prompt") or str(example)
        else:
            text = _apply_chat_template(tokenizer, messages)
        return {"text": text}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # NOTE:
    # - NVFP4 is "true FP4" (W4A4) and requires calibration data.
    # - On GPUs < SM100, vLLM may run weight-only even if activations were calibrated/saved.
    recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_calibration_samples,
    )

    if not args.skip_generate:
        print("\n\n========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
            model.device
        )
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, save_compressed=True)
    tokenizer.save_pretrained(args.save_dir)
    print(f"Saved quantized model to: {args.save_dir}")


if __name__ == "__main__":
    main()
