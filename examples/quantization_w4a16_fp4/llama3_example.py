from __future__ import annotations

import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# NOTE:
# - The 405B checkpoint folder you have locally does not include tokenizer files.
# - Llama 3.1 uses a shared tokenizer across sizes, so we default to your local 8B tokenizer.
DEFAULT_MODEL_PATH = "/home/scratch.mkilaru_coreai/Llama-3.1-405B-Instruct"
DEFAULT_TOKENIZER_PATH = "/home/scratch.mkilaru_coreai/Llama-3.1-8B-Instruct"
DEFAULT_OUTPUT_ROOT = "/home/scratch.mkilaru_coreai"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize Llama 3.1 405B weights to NVFP4A16 (compressed-tensors) using a local checkpoint."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--save-dir", default=None)
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

    # Configure the quantization algorithm and scheme.
    # - Quantize the weights to fp4 (NVFP4) with per group 16 via PTQ.
    recipe = QuantizationModifier(
        targets="Linear", scheme="NVFP4A16", ignore=["lm_head"]
    )

    # Apply quantization.
    oneshot(model=model, recipe=recipe)

    if not args.skip_generate:
        print("\n\n========== SAMPLE GENERATION ==============")
        dispatch_for_generation(model)
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
            model.device
        )
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

    # Save to disk in compressed-tensors format.
    model_name = os.path.basename(os.path.normpath(args.model_path))
    save_dir = args.save_dir or os.path.join(
        DEFAULT_OUTPUT_ROOT, f"{model_name}-NVFP4A16"
    )
    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved quantized model to: {save_dir}")


if __name__ == "__main__":
    main()
