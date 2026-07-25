from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from accelerate import dispatch_model, init_empty_weights
from compressed_tensors import ModelCompressor
from huggingface_hub import save_torch_state_dict
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen3_5MoeForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

DEFAULT_MODEL_PATH = "/home/scratch.mkilaru_coreai/NVIDIA-Ising-Calibration-1-35B-A3B"
DEFAULT_OUTPUT_PATH = (
    "/home/scratch.mkilaru_coreai/NVIDIA-Ising-Calibration-1-35B-A3B-FP8-v6"
)

COPY_THROUGH_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "README.md",
    ".gitattributes",
]


def build_remapped_state_dict(model_path: Path) -> dict[str, torch.Tensor]:
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    unique_shards = []
    seen = set()
    for shard in weight_map.values():
        if shard not in seen:
            seen.add(shard)
            unique_shards.append(shard)

    state: dict[str, torch.Tensor] = {}
    expert_parts: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for shard in unique_shards:
        shard_tensors = load_file(str(model_path / shard))
        for key, tensor in shard_tensors.items():
            parts = key.split(".")
            if (
                len(parts) == 9
                and parts[0] == "model"
                and parts[1] == "language_model"
                and parts[2] == "layers"
                and parts[4] == "mlp"
                and parts[5] == "experts"
                and parts[8] == "weight"
            ):
                layer = parts[3]
                expert = int(parts[6])
                proj = parts[7]
                if proj in {"gate_proj", "up_proj", "down_proj"}:
                    entry = expert_parts.setdefault(layer, {})
                    entry.setdefault(proj, {})[expert] = tensor
                    continue
            state[key] = tensor

    for layer, proj_map in expert_parts.items():
        experts = sorted(set().union(*(d.keys() for d in proj_map.values())))
        gate_up = []
        down = []
        for expert in experts:
            gate = proj_map["gate_proj"][expert]
            up = proj_map["up_proj"][expert]
            down_proj = proj_map["down_proj"][expert]
            gate_up.append(torch.cat([gate, up], dim=0))
            down.append(down_proj)
        state[f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"] = (
            torch.stack(gate_up, dim=0)
        )
        state[f"model.language_model.layers.{layer}.mlp.experts.down_proj"] = (
            torch.stack(down, dim=0)
        )
    return state


def build_manual_device_map(num_layers: int) -> dict[str, int]:
    device_map = {
        "model.visual": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
        "model.language_model.norm": 3,
        "lm_head": 3,
    }
    for i in range(num_layers):
        device_map[f"model.language_model.layers.{i}"] = min(i // 10, 3)
    return device_map


def normalize_saved_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    expert_parts: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for key, value in state_dict.items():
        if key.startswith("model.layers."):
            key = f"model.language_model.{key[len('model.') :]}"
        parts = key.split(".")
        if (
            len(parts) == 9
            and parts[0] == "model"
            and parts[1] == "language_model"
            and parts[2] == "layers"
            and parts[4] == "mlp"
            and parts[5] == "experts"
            and parts[8] == "weight"
        ):
            layer = parts[3]
            expert = int(parts[6])
            proj = parts[7]
            if proj in {"gate_proj", "up_proj", "down_proj"}:
                entry = expert_parts.setdefault(layer, {})
                entry.setdefault(proj, {})[expert] = value
                continue
        normalized[key] = value

    for layer, proj_map in expert_parts.items():
        experts = sorted(set().union(*(d.keys() for d in proj_map.values())))
        gate_up = []
        down = []
        for expert in experts:
            gate_up.append(
                torch.cat(
                    [proj_map["gate_proj"][expert], proj_map["up_proj"][expert]],
                    dim=0,
                )
            )
            down.append(proj_map["down_proj"][expert])
        normalized[
            f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
        ] = torch.stack(gate_up, dim=0)
        normalized[
            f"model.language_model.layers.{layer}.mlp.experts.down_proj"
        ] = torch.stack(down, dim=0)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export NVIDIA-Ising-Calibration-1-35B-A3B to a conservative FP8 checkpoint.\n"
            "Text-side Linear layers are quantized to FP8 with dynamic activations.\n"
            "Vision modules remain in BF16."
        )
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--save-dir", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"--model-path not found: {args.model_path}")

    model_path = Path(args.model_path)
    save_dir = Path(args.save_dir)

    cfg = AutoConfig.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    state_dict = build_remapped_state_dict(model_path)
    with init_empty_weights():
        model = Qwen3_5MoeForConditionalGeneration(cfg)
    model = model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Remapped source load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    # Conservative FP8: keep the full vision stack, routed experts, and MoE
    # routers in BF16. Quantize the rest of the language model, including the
    # shared expert MLP and attention projections.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "re:.*lm_head",
            "re:.*embed_tokens.*",
            "re:.*visual.*",
            "re:.*vision.*",
            "re:.*multi_modal_projector.*",
            "re:.*mlp\\.gate$",
            "re:.*mlp\\.shared_expert_gate$",
            "re:.*linear_attn.in_proj_a$",
            "re:.*linear_attn.in_proj_b$",
        ],
    )

    oneshot(model=model, recipe=recipe)

    if not args.skip_generate:
        print("\n\n========== SAMPLE GENERATION ==============")
        processor = AutoProcessor.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        device_map = build_manual_device_map(cfg.num_hidden_layers)
        model = dispatch_model(model, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        sample = tokenizer(
            "In two concise sentences, explain what an Ising model is.",
            return_tensors="pt",
        )
        sample = {key: value.to("cuda:0") for key, value in sample.items()}
        output = model.generate(**sample, max_new_tokens=args.max_new_tokens)
        print(tokenizer.decode(output[0]))
        print("==========================================\n\n")

    compressor = ModelCompressor.from_pretrained_model(model)
    compressor.compress_model(model)

    state_dict = normalize_saved_state_dict(model.state_dict())
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the config first, then write the quantized state_dict directly so
    # the shard index preserves the in-memory tensor names exactly.
    model.config.save_pretrained(str(save_dir))
    save_torch_state_dict(
        state_dict,
        save_directory=str(save_dir),
        safe_serialization=True,
        max_shard_size="20GB",
    )
    compressor.update_config(str(save_dir))

    # Preserve the original multimodal processor/tokenizer metadata verbatim.
    for name in COPY_THROUGH_FILES:
        src = model_path / name
        if src.exists():
            shutil.copy2(src, save_dir / name)

    print(f"Saved quantized model to: {save_dir}")


if __name__ == "__main__":
    main()
