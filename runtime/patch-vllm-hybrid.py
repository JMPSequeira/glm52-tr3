#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Integrate the hybrid loader and B12X DCP workspace fix into pinned vLLM."""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(old) != 1:
        raise RuntimeError(f"expected exactly one anchor in {path}: {old!r}")
    path.write_text(source.replace(old, new))


def patch(root: Path) -> None:
    registry = root / "model_executor/layers/quantization/__init__.py"
    replace_once(
        registry,
        '    "nvfp4_nf3_hybrid",\n',
        '    "nvfp4_nf3_hybrid",\n    "nvfp4_tr3_hybrid",\n',
    )
    replace_once(
        registry,
        "    from .nvfp4_nf3_hybrid import NvFp4Nf3HybridConfig\n",
        "    from .nvfp4_nf3_hybrid import NvFp4Nf3HybridConfig\n"
        "    from .nvfp4_tr3_hybrid import NvFp4Tr3HybridConfig\n",
    )
    replace_once(
        registry,
        '        "nvfp4_nf3_hybrid": NvFp4Nf3HybridConfig,\n',
        '        "nvfp4_nf3_hybrid": NvFp4Nf3HybridConfig,\n'
        '        "nvfp4_tr3_hybrid": NvFp4Tr3HybridConfig,\n',
    )

    quantization = root / "config/quantization.py"
    replace_once(
        quantization,
        '        "nvfp4_nf3_hybrid",\n',
        '        "nvfp4_nf3_hybrid",\n        "nvfp4_tr3_hybrid",\n',
    )

    model_config = root / "config/model.py"
    model_source = model_config.read_text()
    exl3_anchor = (
        "                # Rank-sliced EXL3 checkpoints retain a ModelOpt dispatch tag\n"
        "                # for backward compatibility, so EXL3 must inspect metadata\n"
        "                # before the ModelOpt overrides claim them.\n"
        '                "exl3",\n'
    )
    if exl3_anchor in model_source:
        model_anchor = exl3_anchor
        model_replacement = (
            "                # Mixed NVFP4/TR3 checkpoints carry the EXL3 marker;\n"
            "                # select the composite backend before pure EXL3.\n"
            '                "nvfp4_tr3_hybrid",\n'
            + exl3_anchor
        )
    else:
        model_anchor = '                "nvfp4_nf3_hybrid",\n'
        model_replacement = '                "nvfp4_tr3_hybrid",\n' + model_anchor
    replace_once(model_config, model_anchor, model_replacement)

    deepseek = root / "model_executor/models/deepseek_v2.py"
    replace_once(
        deepseek,
        "        params_dict = dict(self.named_parameters())\n"
        "        loaded_params: set[str] = set()\n",
        "        params_dict = dict(self.named_parameters())\n"
        "        loaded_params: set[str] = set()\n"
        "        serialized_expert_consumers = {\n"
        "            module_name: module\n"
        "            for module_name, module in self.named_modules()\n"
        '            if hasattr(module, "load_serialized_expert_weight")\n'
        "        }\n",
    )
    replace_once(
        deepseek,
        "            if spec_layer is not None:\n"
        "                continue  # skip spec decode layers for main model\n\n"
        '            if ".indexer." in name and (\n',
        "            if spec_layer is not None:\n"
        "                continue  # skip spec decode layers for main model\n"
        '            if ".experts." in name:\n'
        '                experts_name = name.split(".experts.", 1)[0] + ".experts"\n'
        "                consumer = serialized_expert_consumers.get(\n"
        "                    experts_name\n"
        "                ) or serialized_expert_consumers.get(\n"
        '                    experts_name + ".routed_experts"\n'
        "                )\n"
        "                if consumer is not None and consumer.load_serialized_expert_weight(\n"
        "                    name, loaded_weight\n"
        "                ):\n"
        "                    continue\n\n"
        '            if ".indexer." in name and (\n',
    )

    routed = root / "model_executor/layers/fused_moe/routed_experts.py"
    replace_once(
        routed,
        "    def load_weights(\n",
        "    def load_serialized_expert_weight(\n"
        "        self, name: str, loaded_weight: torch.Tensor\n"
        "    ) -> bool:\n"
        '        loader = getattr(self.quant_method, "load_serialized_expert_weight", None)\n'
        "        if loader is None:\n"
        "            return False\n"
        "        return bool(loader(self, name, loaded_weight))\n\n"
        "    def load_weights(\n",
    )

    sparse_mla = root / "v1/attention/backends/mla/b12x_mla_sparse.py"
    replace_once(
        sparse_mla,
        "        expected_attn_stride = (\n"
        "            self.kv_lora_rank,\n"
        "            (self._max_batched if self._pad_heads else num_tokens) * self.kv_lora_rank,\n"
        "            1,\n"
        "        )\n"
        "        if (\n"
        "            tuple(attn_out.shape)\n"
        "            != (num_tokens, self._input_num_heads, self.kv_lora_rank)\n"
        "            or tuple(attn_out.stride()) != expected_attn_stride\n",
        "        # The B12X plan may round its head-major output pitch above both\n"
        "        # num_tokens and vLLM's speculative-decode-adjusted batch limit\n"
        "        # (for example, 2,048 rows for a 2,047-token MTP prefill chunk).\n"
        "        # Projection compacts this view before cuBLAS, so accept any\n"
        "        # integral head pitch that covers all visible token rows.\n"
        "        attn_stride = tuple(attn_out.stride())\n"
        "        min_head_pitch = num_tokens * self.kv_lora_rank\n"
        "        valid_attn_stride = (\n"
        "            len(attn_stride) == 3\n"
        "            and attn_stride[0] == self.kv_lora_rank\n"
        "            and attn_stride[1] >= min_head_pitch\n"
        "            and attn_stride[1] % self.kv_lora_rank == 0\n"
        "            and attn_stride[2] == 1\n"
        "        )\n"
        "        if (\n"
        "            tuple(attn_out.shape)\n"
        "            != (num_tokens, self._input_num_heads, self.kv_lora_rank)\n"
        "            or not valid_attn_stride\n",
    )
    replace_once(
        sparse_mla,
        '                f"{tuple(attn_out.stride())}, expected stride="\n'
        '                f"{expected_attn_stride}"\n',
        '                f"{attn_stride}, required minimum head pitch="\n'
        '                f"{min_head_pitch}"\n',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="path to the vllm Python package")
    args = parser.parse_args()
    patch(args.root)


if __name__ == "__main__":
    main()
