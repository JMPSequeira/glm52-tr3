# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Mixed NVFP4 + TR3 trellis MoE quantization for GLM-5.2.

The hot expert tier uses ModelOpt NVFP4 and the B12X W4A16 fused-MoE path.  The
remaining experts use the checkpoint's TP4-pre-sliced ExLlamaV3 trellis format.
Tier metadata is loaded from ``tier_bitmap.json`` and injected into the normal
vLLM quantization config; no import hooks or global model-loader patches are
used.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.nvfp4_nf3_hybrid import (
    NvFp4Nf3HybridConfig,
    NvFp4Nf3HybridMoEMethod,
    _HybridLayerState,
    _HybridSharedRuntime,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe import RoutedExperts, SharedExperts

logger = init_logger(__name__)

_TR3_TENSOR_RE = re.compile(
    r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\."
    r"rank(\d+)\.(trellis|suh|svh|mcg)$"
)
_TR3_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_TR3_FIELDS = ("trellis", "suh", "svh")

def _tile_config_env(name: str, default: str) -> tuple[int, int, int, int]:
    values = tuple(int(part.strip()) for part in os.getenv(name, default).split(","))
    if len(values) != 4 or any(value <= 0 or value % 16 for value in values):
        raise ValueError(
            f"{name} must contain four comma-separated positive multiples of 16"
        )
    return values


_T256_ENABLED = os.getenv("B12X_TRELLIS256_MOE", "0") == "1"
_MIXED_TRELLIS_ENABLED = os.getenv("VLLM_TR3_MIXED_PERSISTENT", "0") == "1"
_MIXED_TRELLIS_MAX_M = max(
    1, int(os.getenv("VLLM_TR3_MIXED_PERSISTENT_MAX_M", "4"))
)
_MIXED_TRELLIS_TILE_CONFIG = _tile_config_env(
    "VLLM_TR3_MIXED_PERSISTENT_TILE_CONFIG", "128,128,128,128"
)
_MIXED_TRELLIS_PARITY = os.getenv("VLLM_TR3_MIXED_PERSISTENT_PARITY", "0") == "1"
_MIXED_TRELLIS_PARITY_REL_MAX = float(
    os.getenv("VLLM_TR3_MIXED_PERSISTENT_PARITY_REL_MAX", "0.01")
)
_MIXED_TRELLIS_PARITY_COS_MIN = float(
    os.getenv("VLLM_TR3_MIXED_PERSISTENT_PARITY_COS_MIN", "0.99995")
)
_PLANNED_TAIL_ENABLED = os.getenv("VLLM_TR3_PLANNED_TAIL", "0") == "1"
_PLANNED_TAIL_PARITY = os.getenv("VLLM_TR3_PLANNED_TAIL_PARITY", "0") == "1"
_PLANNED_TAIL_OVERLAP = os.getenv("VLLM_TR3_PLANNED_TAIL_OVERLAP", "0") == "1"
_PLANNED_TAIL_PARITY_REL_MAX = float(
    os.getenv("VLLM_TR3_PLANNED_TAIL_PARITY_REL_MAX", "0.005")
)
_PLANNED_TAIL_PARITY_COS_MIN = float(
    os.getenv("VLLM_TR3_PLANNED_TAIL_PARITY_COS_MIN", "0.99999")
)
_PLANNED_TAIL_MIN_M = max(1, int(os.getenv("VLLM_TR3_PLANNED_TAIL_MIN_M", "1")))
_PLANNED_TAIL_MAX_M = max(
    _PLANNED_TAIL_MIN_M,
    int(os.getenv("VLLM_TR3_PLANNED_TAIL_MAX_M", "32")),
)
_PLANNED_TAIL_BLOCK_M = int(os.getenv("VLLM_TR3_PLANNED_TAIL_BLOCK_M", "8"))
_PLANNED_PREFILL_ENABLED = os.getenv("VLLM_TR3_PLANNED_PREFILL", "0") == "1"
_PLANNED_PREFILL_MAX_M = max(
    _PLANNED_TAIL_MAX_M,
    int(os.getenv("VLLM_TR3_PLANNED_PREFILL_MAX_M", "6144")),
)
_PLANNED_PREFILL_BLOCK_M = int(
    os.getenv("VLLM_TR3_PLANNED_PREFILL_BLOCK_M", "64")
)
_PLANNED_PREFILL_PARITY_MAX_M = int(
    os.getenv("VLLM_TR3_PLANNED_PREFILL_PARITY_MAX_M", "128")
)
_PLANNED_PREFILL_PARITY_REL_MAX = float(
    os.getenv("VLLM_TR3_PLANNED_PREFILL_PARITY_REL_MAX", "0.01")
)
_PLANNED_PREFILL_PARITY_COS_MIN = float(
    os.getenv("VLLM_TR3_PLANNED_PREFILL_PARITY_COS_MIN", "0.99995")
)
_T256_MIN_M = max(1, int(os.getenv("B12X_TRELLIS256_MIN_M", "12")))
_T256_MAX_M = max(
    _T256_MIN_M, int(os.getenv("B12X_TRELLIS256_MAX_M", "16"))
)
_T256_TILE_CONFIG = (64, 256, 64, 256)


class _Tr3SharedRuntime(_HybridSharedRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.tr3_max_m: int | None = None
        self.tr3_topk: int | None = None
        self.tr3_num_experts: int | None = None
        self.tr3_cap = int(os.getenv("B12X_TR3_MAX_TOKENS_PER_EXPERT", "128"))
        self.tr3_x: torch.Tensor | None = None
        self.tr3_output: torch.Tensor | None = None
        self.tr3_temp_gate: torch.Tensor | None = None
        self.tr3_temp_up: torch.Tensor | None = None
        self.tr3_temp_intermediate_gate: torch.Tensor | None = None
        self.tr3_temp_intermediate_up: torch.Tensor | None = None
        self.tr3_flat_token: torch.Tensor | None = None
        self.tr3_ones: torch.Tensor | None = None
        self.tr3_expert_count: torch.Tensor | None = None
        self.tr3_expert_offsets: torch.Tensor | None = None
        self.tr3_token_sorted: torch.Tensor | None = None
        self.tr3_weight_sorted: torch.Tensor | None = None
        self.t256_launch: Any | None = None
        self.t256_buffers: Any | None = None
        self.t256_route_capacity: int | None = None
        self.t256_topk: int | None = None
        self.t256_dummy_scale: torch.Tensor | None = None
        self.t256_ones: torch.Tensor | None = None
        self.t256_route_token: torch.Tensor | None = None
        self.t256_identity_emap: torch.Tensor | None = None
        self.t256_seen_m: set[tuple[int, int, int]] = set()
        self.t256_prepared_layers: set[str] = set()
        self.t256_cache_released = False
        self.planned_api: Any | None = None
        self.planned_plan: Any | None = None
        self.planned_scratch: torch.Tensor | None = None
        self.planned_prefill_plan: Any | None = None
        self.planned_prefill_scratch: torch.Tensor | None = None
        self.planned_topk: int | None = None
        self.planned_prepared_layers: set[str] = set()
        self.planned_cache_released = False
        self.planned_stream: torch.cuda.Stream | None = None
        self.mixed_launch: Any | None = None
        self.mixed_fc1: torch.Tensor | None = None
        self.mixed_activated: torch.Tensor | None = None
        self.mixed_route_output: torch.Tensor | None = None
        self.mixed_output: torch.Tensor | None = None
        self.mixed_fc1_tmp: torch.Tensor | None = None
        self.mixed_fc2_tmp: torch.Tensor | None = None
        self.mixed_workspace: torch.Tensor | None = None
        self.mixed_rotation_a_gate: torch.Tensor | None = None
        self.mixed_rotation_a_up: torch.Tensor | None = None
        self.mixed_topk: int | None = None


def _inject_tr3_tiers(
    hf_quant_cfg: dict[str, Any], hf_config: Any
) -> tuple[dict[str, list[int]], dict[str, Any]] | None:
    """Resolve ``tier_bitmap.json`` and add its compact bit map to config."""
    if hf_config is None:
        return None
    metadata = getattr(hf_config, "hybrid_tr3_tail", None)
    if not isinstance(metadata, dict):
        return None
    if metadata.get("format") != "exl3-trellis":
        raise ValueError(
            f"Unsupported hybrid_tr3_tail format: {metadata.get('format')}"
        )

    model = getattr(hf_config, "_name_or_path", None)
    if not model:
        raise ValueError("TR3 checkpoint config does not identify its model repository")
    revision = getattr(hf_config, "_commit_hash", None) or "main"
    tier_bitmap = get_hf_file_to_dict("tier_bitmap.json", model, revision)
    if not isinstance(tier_bitmap, dict) or not tier_bitmap:
        raise ValueError(f"Could not load tier_bitmap.json for TR3 checkpoint {model}")

    num_experts = int(metadata["experts_per_layer"])
    expected_kept = int(metadata["nvfp4_keep_per_layer"])
    bit_map: dict[str, list[int]] = {}
    for layer_index, entry in tier_bitmap.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("keep_nvfp4"), list):
            raise ValueError(f"Invalid tier_bitmap entry for layer {layer_index}")
        kept = {int(expert) for expert in entry["keep_nvfp4"]}
        if len(kept) != expected_kept or any(
            expert < 0 or expert >= num_experts for expert in kept
        ):
            raise ValueError(
                f"Invalid TR3 kept tier for layer {layer_index}: {len(kept)} experts"
            )
        bit_map[str(layer_index)] = [
            4 if expert in kept else 3 for expert in range(num_experts)
        ]

    injected = {
        "hybrid_bit_map": bit_map,
        "hybrid_tr3_tail": dict(metadata),
    }
    hf_quant_cfg.update(injected)
    config_quant = getattr(hf_config, "quantization_config", None)
    if isinstance(config_quant, dict):
        config_quant.update(injected)
    return bit_map, metadata


class NvFp4Tr3HybridConfig(NvFp4Nf3HybridConfig):
    """ModelOpt NVFP4 hot experts plus a TP4-native TR3 tail."""

    def get_name(self) -> QuantizationMethods:
        return "nvfp4_tr3_hybrid"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> QuantizationMethods | None:
        if user_quant is not None and user_quant != "nvfp4_tr3_hybrid":
            return None
        if not isinstance(hf_quant_cfg, dict):
            return None
        if _inject_tr3_tiers(hf_quant_cfg, hf_config) is not None:
            return "nvfp4_tr3_hybrid"
        return None

    @classmethod
    def _from_config(
        cls,
        *,
        quant_method: str,
        kv_cache_quant_method: str | None,
        exclude_modules: list[str],
        original_config: dict[str, Any],
        group_size: int | None,
        **kwargs: Any,
    ) -> NvFp4Tr3HybridConfig:
        config = super()._from_config(
            quant_method=quant_method,
            kv_cache_quant_method=kv_cache_quant_method,
            exclude_modules=exclude_modules,
            original_config=original_config,
            group_size=group_size,
            **kwargs,
        )
        assert isinstance(config, NvFp4Tr3HybridConfig)
        metadata = original_config.get("hybrid_tr3_tail")
        if not isinstance(metadata, dict):
            raise ValueError("nvfp4_tr3_hybrid requires hybrid_tr3_tail metadata")
        if int(metadata.get("tp", 0)) != 4:
            raise ValueError(
                "TR3 checkpoint is pre-sliced for "
                f"TP={metadata.get('tp')}; only TP4 is supported"
            )
        config.tr3_metadata = metadata
        config.shared_runtime = _Tr3SharedRuntime()
        return config


class NvFp4Tr3HybridMoEMethod(NvFp4Nf3HybridMoEMethod):
    """Compact NVFP4/TR3 expert loader and two-tier fused forward."""

    quant_config: NvFp4Tr3HybridConfig

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype, extra_weight_attrs
        if not self.quant_config.is_checkpoint_nvfp4_serialized:
            raise ValueError("TR3 requires a serialized ModelOpt NVFP4 checkpoint")
        if layer.activation is not MoEActivation.SILU:
            raise NotImplementedError(
                "nvfp4_tr3_hybrid only supports SiLU MoE layers, "
                f"got {layer.activation}"
            )
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != int(self.quant_config.tr3_metadata["tp"]):
            raise ValueError(f"TR3 checkpoint requires TP4, got TP{tp_size}")

        bits = self._layer_bits(layer)
        if bits is None:
            bits = [4] * num_experts
        if len(bits) != num_experts:
            raise ValueError(
                f"TR3 tier map for {layer.layer_name} has {len(bits)} experts; "
                f"expected {num_experts}"
            )
        kept = [expert for expert, width in enumerate(bits) if width == 4]
        tail = [expert for expert, width in enumerate(bits) if width == 3]
        if len(kept) + len(tail) != num_experts:
            raise ValueError(
                f"TR3 tier map for {layer.layer_name} contains invalid widths"
            )
        if tail:
            expected_kept = int(self.quant_config.tr3_metadata["nvfp4_keep_per_layer"])
            expected_tail = int(self.quant_config.tr3_metadata["tr3_tail_per_layer"])
            if (len(kept), len(tail)) != (expected_kept, expected_tail):
                raise ValueError(
                    f"TR3 tier sizes for {layer.layer_name} are "
                    f"{len(kept)}/{len(tail)}; "
                    f"expected {expected_kept}/{expected_tail}"
                )

        remap = {
            **{expert: (0, local) for local, expert in enumerate(kept)},
            **{expert: (1, local) for local, expert in enumerate(tail)},
        }
        state = _HybridLayerState(
            remap, hidden_size, intermediate_size_per_partition, num_experts, False
        )
        state.tr3_slabs = {}
        state.tr3_loaded = set()
        state.tr3_pointer_tables = None
        state.tr3_emap = None
        state.tr3_runtime_ready = False
        state.t256_w13_backing = None
        state.t256_rotation = None
        state.t256_prep = None
        state.t256_emap = None
        state.t256_gate_suh = None
        state.t256_down_svh = None
        state.planned_weights = None
        state.planned_route_map = None
        state.planned_output_map = None
        state.planned_parity_checked = False
        state.planned_prefill_parity_checked = False
        state.planned_fork_event = None
        state.planned_done_event = None
        state.mixed_tier_map = None
        state.mixed_parity_checked = False
        layer.hybrid_state = state

        tp_rank = get_tensor_model_parallel_rank()
        group_size = self.quant_config.group_size
        hidden = hidden_size
        inter = intermediate_size_per_partition

        def hot_weight_loader(
            param: torch.nn.Parameter,
            loaded_weight: torch.Tensor,
            name_mapped: str | None = None,
            *,
            weight_name: str | None = None,
            shard_id: str | None = None,
            expert_id: int | None = None,
            return_success: bool = False,
            **kwargs,
        ) -> bool:
            del param, return_success, kwargs
            name = name_mapped or weight_name or ""
            if "input_scale" in name:
                return True
            if expert_id is None:
                raise ValueError(f"Missing expert id while loading {name}")
            tier, local_id = state.remap[int(expert_id)]
            if tier != 0:
                raise ValueError(
                    f"TR3 tail tensor was routed through NVFP4 loader: {name}"
                )
            family = "w13" if "w13_" in name else "w2"
            if "weight_scale_2" in name:
                target = getattr(layer, f"{family}_weight_scale_2")
                value = loaded_weight.reshape(()).to(target.dtype)
                if family == "w13":
                    target.data[local_id, 0 if shard_id == "w1" else 1] = value
                else:
                    target.data[local_id] = value
                return True
            if loaded_weight.ndim >= 2:
                if shard_id in ("w1", "w3"):
                    loaded_weight = loaded_weight.chunk(tp_size, 0)[tp_rank]
                elif shard_id == "w2":
                    loaded_weight = loaded_weight.chunk(tp_size, 1)[tp_rank]
            if "weight_scale" in name:
                target = getattr(layer, f"{family}_nv_scale")
            else:
                target = getattr(layer, f"{family}_weight")
            destination = target.data[local_id]
            if family == "w13" and shard_id in ("w1", "w3"):
                half = destination.shape[0] // 2
                destination = (
                    destination[:half] if shard_id == "w1" else destination[half:]
                )
            destination.copy_(
                loaded_weight.reshape(destination.shape).to(destination.dtype)
            )
            return True

        def register_dispatcher(
            name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8
        ) -> None:
            parameter = torch.nn.Parameter(
                torch.zeros(shape, dtype=dtype, device=torch.cuda.current_device()),
                requires_grad=False,
            )
            set_weight_attrs(parameter, {"weight_loader": hot_weight_loader})
            layer.register_parameter(name, parameter)

        num_kept = max(state.num_kept, 1)
        register_dispatcher("w13_weight", (num_kept, 2 * inter, hidden // 2))
        register_dispatcher("w13_weight_scale", (1,))
        register_dispatcher("w13_weight_scale_2", (num_kept, 2), torch.float32)
        register_dispatcher("w13_input_scale", (1,), torch.float32)
        register_dispatcher("w2_weight", (num_kept, hidden, inter // 2))
        register_dispatcher("w2_weight_scale", (1,))
        register_dispatcher("w2_weight_scale_2", (num_kept,), torch.float32)
        register_dispatcher("w2_input_scale", (1,), torch.float32)
        for name, shape in (
            ("w13_nv_scale", (num_kept, 2 * inter, hidden // group_size)),
            ("w2_nv_scale", (num_kept, hidden, inter // group_size)),
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(
                        shape,
                        dtype=torch.float8_e4m3fn,
                        device=torch.cuda.current_device(),
                    ),
                    requires_grad=False,
                ),
            )

        if state.num_nf3:
            tail_count = state.num_nf3
            shapes = {
                ("gate_proj", "trellis"): (
                    tail_count,
                    hidden // 16,
                    inter // 16,
                    48,
                ),
                ("gate_proj", "suh"): (tail_count, hidden),
                ("gate_proj", "svh"): (tail_count, inter),
                ("up_proj", "trellis"): (
                    tail_count,
                    hidden // 16,
                    inter // 16,
                    48,
                ),
                ("up_proj", "suh"): (tail_count, hidden),
                ("up_proj", "svh"): (tail_count, inter),
                ("down_proj", "trellis"): (
                    tail_count,
                    inter // 16,
                    hidden // 16,
                    48,
                ),
                ("down_proj", "suh"): (tail_count, inter),
                ("down_proj", "svh"): (tail_count, hidden),
            }
            shared_fc1 = None
            if _T256_ENABLED or _PLANNED_TAIL_ENABLED:
                shared_fc1 = torch.empty(
                    (2,) + shapes[("gate_proj", "trellis")],
                    dtype=torch.int16,
                    device=torch.cuda.current_device(),
                )
                state.t256_w13_backing = shared_fc1
                state.t256_rotation = torch.empty(
                    (tail_count, 3 * inter),
                    dtype=torch.float16,
                    device=torch.cuda.current_device(),
                )
            for (projection, field), shape in shapes.items():
                if (
                    shared_fc1 is not None
                    and field == "trellis"
                    and projection in ("gate_proj", "up_proj")
                ):
                    tensor = shared_fc1[0 if projection == "gate_proj" else 1]
                else:
                    tensor = torch.empty(
                        shape,
                        dtype=torch.int16 if field == "trellis" else torch.float16,
                        device=torch.cuda.current_device(),
                    )
                layer.register_buffer(
                    f"tr3_{projection}_{field}", tensor, persistent=False
                )
                state.tr3_slabs[(projection, field)] = tensor

    def load_serialized_expert_weight(
        self, layer: RoutedExperts, name: str, loaded_weight: torch.Tensor
    ) -> bool:
        """Consume one checkpoint-native rank-sliced trellis tensor."""
        match = _TR3_TENSOR_RE.search(name)
        if match is None:
            return False
        expert_id = int(match.group(1))
        projection = match.group(2)
        checkpoint_rank = int(match.group(3))
        field = match.group(4)
        state: _HybridLayerState = layer.hybrid_state
        tier, local_id = state.remap[expert_id]
        if tier != 1:
            raise ValueError(f"TR3 tensor targets kept expert {expert_id}: {name}")
        if checkpoint_rank != get_tensor_model_parallel_rank():
            return True

        key = (expert_id, projection, field)
        if key in state.tr3_loaded:
            raise ValueError(f"Duplicate TR3 tensor {name}")
        if field == "mcg":
            expected = int(self.quant_config.tr3_metadata["mcg_multiplier"])
            actual = int(loaded_weight.reshape(()).item()) & 0xFFFFFFFF
            if actual != expected:
                raise ValueError(f"TR3 codebook mismatch in {name}: {actual:#x}")
        else:
            destination = state.tr3_slabs[(projection, field)][local_id]
            expected_dtype = torch.int16 if field == "trellis" else torch.float16
            if loaded_weight.dtype != expected_dtype:
                raise ValueError(
                    f"TR3 tensor {name} has dtype {loaded_weight.dtype}; "
                    f"expected {expected_dtype}"
                )
            if loaded_weight.numel() != destination.numel():
                raise ValueError(
                    f"TR3 tensor {name} has {loaded_weight.numel()} values; "
                    f"expected {destination.numel()}"
                )
            destination.copy_(
                loaded_weight.reshape(destination.shape), non_blocking=True
            )
        state.tr3_loaded.add(key)
        return True

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        from b12x.moe.fused.w4a16.prepare import (
            W4A16PackedWeights,
            _make_workspace,
            _permute_nvfp4_scales,
            _repack_weight,
        )

        state: _HybridLayerState = layer.hybrid_state
        hidden, inter = state.hidden_size, state.intermediate_size
        device = layer.w13_weight.device
        num_kept = state.num_kept
        emap_kept = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        for global_id, (tier, local_id) in state.remap.items():
            if tier == 0:
                emap_kept[global_id] = local_id
        state.emap_kept = emap_kept

        if num_kept:
            global_w13 = layer.w13_weight_scale_2[:num_kept, 0].contiguous()
            global_w2 = layer.w2_weight_scale_2[:num_kept].contiguous()
            packed_w13 = _repack_weight(
                layer.w13_weight.contiguous(), size_k=hidden, size_n=2 * inter
            )
            packed_w2 = _repack_weight(
                layer.w2_weight.contiguous(), size_k=inter, size_n=hidden
            )
            scale_w13, global_w13 = _permute_nvfp4_scales(
                layer.w13_nv_scale,
                global_w13,
                size_k=hidden,
                size_n=2 * inter,
                a_dtype=torch.bfloat16,
            )
            scale_w2, global_w2 = _permute_nvfp4_scales(
                layer.w2_nv_scale,
                global_w2,
                size_k=inter,
                size_n=hidden,
                a_dtype=torch.bfloat16,
            )
            state.prep_kept = W4A16PackedWeights(
                w13=packed_w13,
                w13_scale=scale_w13,
                w13_global_scale=global_w13,
                w2=packed_w2,
                w2_scale=scale_w2,
                w2_global_scale=global_w2,
                workspace=_make_workspace(device),
                hidden_size=hidden,
                intermediate_size=inter,
                num_experts=num_kept,
                is_gated=True,
                params_dtype=torch.bfloat16,
                source_format="modelopt_nvfp4",
                w13_layout="w13",
                weight_layout="packed",
                scale_format="e4m3_k16",
            )
            for name in ("w13_weight", "w2_weight", "w13_nv_scale", "w2_nv_scale"):
                parameter = getattr(layer, name)
                parameter.data = parameter.data.new_empty((0,))

        if state.num_nf3:
            expected = state.num_nf3 * len(_TR3_PROJECTIONS) * 4
            if len(state.tr3_loaded) != expected:
                raise ValueError(
                    f"Incomplete TR3 layer {layer.layer_name}: loaded "
                    f"{len(state.tr3_loaded)}/{expected} local tensors"
                )
            pointer_tables = []
            for projection in _TR3_PROJECTIONS:
                for field in _TR3_FIELDS:
                    slab = state.tr3_slabs[(projection, field)]
                    offsets = torch.arange(
                        state.num_nf3, dtype=torch.int64, device=device
                    )
                    pointer_tables.append(
                        offsets * (slab.stride(0) * slab.element_size())
                        + slab.data_ptr()
                    )
            state.tr3_pointer_tables = tuple(pointer_tables)
            emap = torch.full(
                (state.num_experts,),
                state.num_nf3,
                dtype=torch.int64,
                device=device,
            )
            for global_id, (tier, local_id) in state.remap.items():
                if tier == 1:
                    emap[global_id] = local_id
            state.tr3_emap = emap
            logger.info(
                "TR3 layer %s ready: %d NVFP4 + %d trellis experts",
                layer.layer_name,
                state.num_kept,
                state.num_nf3,
            )

    def _build_planned_trellis_moe(
        self, layer: RoutedExperts, topk: int
    ) -> None:
        """Prepare full-rotation tail weights and shared decode/prefill plans."""
        import gc

        from sparkinfer.moe import trellis_moe

        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        if state.planned_weights is not None:
            if runtime.planned_topk != topk:
                raise RuntimeError("planned Trellis top-k changed after preparation")
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "planned Trellis weights and launch must be built before graph capture"
            )

        tail = state.num_nf3
        hidden = state.hidden_size
        inter = state.intermediate_size
        backing = state.t256_w13_backing
        rotation = state.t256_rotation
        if backing is None or rotation is None:
            raise RuntimeError("planned Trellis pre-load backing is missing")

        gate_svh = state.tr3_slabs[("gate_proj", "svh")]
        up_svh = state.tr3_slabs[("up_proj", "svh")]
        down_suh = state.tr3_slabs[("down_proj", "suh")]
        rotation[:, :inter].copy_(gate_svh)
        rotation[:, inter : 2 * inter].copy_(up_svh)
        rotation[:, 2 * inter :].copy_(down_suh)
        state.planned_weights = trellis_moe.prepare_weights(
            backing,
            state.tr3_slabs[("down_proj", "trellis")],
            gate_suh=state.tr3_slabs[("gate_proj", "suh")],
            up_suh=state.tr3_slabs[("up_proj", "suh")],
            intermediate_rotations=rotation,
            down_svh=state.tr3_slabs[("down_proj", "svh")],
            codebook="mcg",
            mcg=int(self.quant_config.tr3_metadata["mcg_multiplier"]),
            tile_config=_T256_TILE_CONFIG,
        )
        route_map = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=backing.device
        )
        output_map = torch.full_like(route_map, -1)
        for global_id, (tier, local_id) in state.remap.items():
            if tier == 1:
                route_map[global_id] = local_id
                output_map[global_id] = local_id
        state.planned_route_map = route_map
        state.planned_output_map = output_map
        if _PLANNED_TAIL_OVERLAP:
            state.planned_fork_event = torch.cuda.Event()
            state.planned_done_event = torch.cuda.Event()

        if runtime.planned_plan is None:
            def plan_with_scratch(max_tokens: int, block_m: int):
                caps = trellis_moe.Caps(
                    max_tokens=max_tokens,
                    num_topk=topk,
                    num_experts=tail,
                    hidden_size=hidden,
                    intermediate_size=inter,
                    route_num_experts=state.num_experts,
                    block_size_m=block_m,
                    trellis_bits=3,
                    tile_config=_T256_TILE_CONFIG,
                    input_dtype=torch.bfloat16,
                    device=backing.device,
                )
                plan = trellis_moe.plan(caps)
                scratch_specs = plan.scratch_specs()
                if len(scratch_specs) != 1:
                    raise RuntimeError("planned Trellis expected one scratch arena")
                spec = scratch_specs[0]
                scratch = torch.empty(
                    spec.shape, dtype=spec.dtype, device=spec.device
                )
                return plan, scratch

            runtime.planned_api = trellis_moe
            runtime.planned_plan, runtime.planned_scratch = plan_with_scratch(
                _PLANNED_TAIL_MAX_M, _PLANNED_TAIL_BLOCK_M
            )
            if _PLANNED_PREFILL_ENABLED:
                (
                    runtime.planned_prefill_plan,
                    runtime.planned_prefill_scratch,
                ) = plan_with_scratch(
                    _PLANNED_PREFILL_MAX_M, _PLANNED_PREFILL_BLOCK_M
                )
            runtime.planned_topk = topk
            if _PLANNED_TAIL_OVERLAP:
                runtime.planned_stream = torch.cuda.Stream(device=backing.device)
            prefill_mib = (
                0.0
                if runtime.planned_prefill_scratch is None
                else runtime.planned_prefill_scratch.numel()
                * runtime.planned_prefill_scratch.element_size()
                / (1 << 20)
            )
            logger.info(
                "Planned Trellis runtime ready: decode m=%d..%d block_m=%d, "
                "prefill=%s max_m=%d block_m=%d arena=%.1fMiB topk=%d",
                _PLANNED_TAIL_MIN_M,
                _PLANNED_TAIL_MAX_M,
                _PLANNED_TAIL_BLOCK_M,
                _PLANNED_PREFILL_ENABLED,
                _PLANNED_PREFILL_MAX_M,
                _PLANNED_PREFILL_BLOCK_M,
                prefill_mib,
                topk,
            )
        elif runtime.planned_topk != topk:
            raise RuntimeError("planned Trellis shared routing geometry changed")

        runtime.planned_prepared_layers.add(layer.layer_name)
        if (
            not runtime.planned_cache_released
            and len(runtime.planned_prepared_layers)
            == len(self.quant_config.hybrid_bit_map)
        ):
            before = torch.cuda.memory_reserved(backing.device)
            gc.collect()
            torch.cuda.empty_cache()
            runtime.planned_cache_released = True
            logger.info(
                "Planned Trellis released eager setup cache: %d -> %d MiB reserved",
                before >> 20,
                torch.cuda.memory_reserved(backing.device) >> 20,
            )

    def _build_mixed_trellis_moe(
        self, layer: RoutedExperts, topk: int
    ) -> None:
        """Prepare one persistent NVFP4/Trellis decode grid."""
        from sparkinfer.moe._shared.kernels.w4a16.host import (
            packed_gemm_scratch_elements,
        )
        from sparkinfer.moe._shared.kernels.w4a16.trellis_hybrid import (
            compile_w4a16_trellis_hybrid,
        )

        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        if state.mixed_tier_map is not None:
            if runtime.mixed_topk != topk:
                raise RuntimeError("mixed Trellis top-k changed after preparation")
            return
        if state.prep_kept is None or state.planned_weights is None:
            raise RuntimeError("mixed Trellis requires prepared NVFP4 and tail weights")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("mixed Trellis must be prepared before graph capture")

        device = state.prep_kept.w13.device
        descriptors = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        for global_id, (tier, local_id) in state.remap.items():
            descriptors[global_id] = (int(tier) << 8) | int(local_id)
        state.mixed_tier_map = descriptors

        if runtime.mixed_launch is None:
            props = torch.cuda.get_device_properties(device)
            sms = int(props.multi_processor_count)

            runtime.mixed_launch = compile_w4a16_trellis_hybrid(
                size_m=_MIXED_TRELLIS_MAX_M,
                hidden_size=state.hidden_size,
                intermediate_size=state.intermediate_size,
                tier0_num_experts=state.num_kept,
                tier1_num_experts=state.num_nf3,
                top_k=topk,
                map_slots=state.num_experts,
                sms=sms,
                max_shared_mem=int(
                    getattr(props, "shared_memory_per_block_optin", 101_376)
                ),
                force_tile_config=_MIXED_TRELLIS_TILE_CONFIG,
                trellis_bits=3,
            )
            routes = _MIXED_TRELLIS_MAX_M * topk
            hidden = state.hidden_size
            inter = state.intermediate_size
            scratch_elements = packed_gemm_scratch_elements(
                size_n=max(2 * inter, hidden),
                route_slots=routes,
                moe_block_size=8,
                sms=sms,
            )
            arena = runtime.planned_scratch
            if (
                arena is None
                or arena.dtype != torch.uint8
                or arena.ndim != 1
                or not arena.is_contiguous()
            ):
                raise RuntimeError(
                    "mixed Trellis requires the contiguous planned decode arena"
                )
            arena_cursor = 0

            def take_arena(
                shape: tuple[int, ...], dtype: torch.dtype
            ) -> torch.Tensor:
                nonlocal arena_cursor
                elements = 1
                for extent in shape:
                    elements *= extent
                item_size = torch.empty((), dtype=dtype).element_size()
                offset = (arena_cursor + 255) & -256
                byte_count = elements * item_size
                end = offset + byte_count
                if end > arena.numel():
                    raise RuntimeError(
                        "planned decode arena is too small for mixed Trellis: "
                        f"requires {end} bytes, has {arena.numel()}"
                    )
                arena_cursor = end
                return (
                    arena.narrow(0, offset, byte_count)
                    .view(dtype)
                    .view(shape)
                )

            runtime.mixed_fc1 = take_arena(
                (routes, 2 * inter), torch.bfloat16
            )
            runtime.mixed_activated = take_arena(
                (routes, inter), torch.bfloat16
            )
            runtime.mixed_route_output = take_arena(
                (routes, hidden), torch.bfloat16
            )
            runtime.mixed_rotation_a_gate = take_arena(
                (routes, hidden), torch.bfloat16
            )
            runtime.mixed_rotation_a_up = take_arena(
                (routes, hidden), torch.bfloat16
            )
            runtime.mixed_fc1_tmp = take_arena(
                (scratch_elements,), torch.float32
            )
            runtime.mixed_fc2_tmp = take_arena(
                (scratch_elements,), torch.float32
            )
            runtime.mixed_workspace = take_arena(
                (sms * 4 + 2,), torch.int32
            )
            runtime.mixed_workspace.zero_()
            # The returned output must survive a parity reference run, which
            # reuses the planned arena immediately after this kernel.
            runtime.mixed_output = torch.empty(
                (_MIXED_TRELLIS_MAX_M, hidden),
                dtype=torch.bfloat16,
                device=device,
            )
            runtime.mixed_topk = topk
            logger.info(
                "Mixed Trellis persistent runtime ready: m=1..%d topk=%d "
                "tiles=%s registers=%d shared=%dB",
                _MIXED_TRELLIS_MAX_M,
                topk,
                _MIXED_TRELLIS_TILE_CONFIG,
                runtime.mixed_launch.registers_per_thread,
                runtime.mixed_launch.shared_memory_bytes,
            )
        elif runtime.mixed_topk != topk:
            raise RuntimeError("mixed Trellis shared geometry changed")

    def _run_mixed_trellis(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from sparkinfer.moe._shared.kernels.w4a16.trellis_hybrid import (
            run_w4a16_trellis_hybrid,
        )

        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        required = (
            runtime.mixed_launch,
            runtime.mixed_fc1,
            runtime.mixed_activated,
            runtime.mixed_route_output,
            runtime.mixed_output,
            runtime.mixed_fc1_tmp,
            runtime.mixed_fc2_tmp,
            runtime.mixed_workspace,
            runtime.mixed_rotation_a_gate,
            runtime.mixed_rotation_a_up,
            state.mixed_tier_map,
            state.prep_kept,
            state.planned_weights,
        )
        if any(value is None for value in required):
            raise RuntimeError("mixed Trellis runtime was not prepared")
        ids = (
            topk_ids
            if topk_ids.dtype == torch.int32
            else topk_ids.to(torch.int32)
        )
        if not ids.is_contiguous():
            ids = ids.contiguous()
        tail = state.planned_weights
        return run_w4a16_trellis_hybrid(
            x,
            state.prep_kept,
            tail._prepared,
            topk_weights,
            ids,
            state.mixed_tier_map,
            launch=runtime.mixed_launch,
            fc1=runtime.mixed_fc1,
            activated=runtime.mixed_activated,
            route_output=runtime.mixed_route_output,
            output=runtime.mixed_output,
            fc1_tmp=runtime.mixed_fc1_tmp,
            fc2_tmp=runtime.mixed_fc2_tmp,
            workspace=runtime.mixed_workspace,
            rotation_a_gate=runtime.mixed_rotation_a_gate,
            rotation_a_up=runtime.mixed_rotation_a_up,
            gate_suh=tail.gate_suh,
            up_suh=tail.up_suh,
            intermediate_rotations=tail.intermediate_rotations,
            down_svh=tail.down_svh,
        )

    def _run_planned_trellis(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        m = int(x.shape[0])
        use_prefill_plan = m > _PLANNED_TAIL_MAX_M
        plan = (
            runtime.planned_prefill_plan
            if use_prefill_plan
            else runtime.planned_plan
        )
        scratch = (
            runtime.planned_prefill_scratch
            if use_prefill_plan
            else runtime.planned_scratch
        )
        if (
            runtime.planned_api is None
            or plan is None
            or scratch is None
            or state.planned_weights is None
            or state.planned_route_map is None
            or state.planned_output_map is None
        ):
            raise RuntimeError("planned Trellis runtime was not prepared")
        binding = runtime.planned_api.bind(
            plan,
            scratch=scratch,
            a=x,
            weights=state.planned_weights,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            route_expert_map=state.planned_route_map,
            output_expert_map=state.planned_output_map,
        )
        return runtime.planned_api.run(binding=binding)

    def _build_trellis256_moe(self, layer: RoutedExperts, topk: int) -> None:
        """Build the native Trellis256 tail over checkpoint-native weights."""
        import dataclasses
        import gc

        from b12x.moe.fused.w4a16.host import (
            make_w4a16_packed_buffers,
            max_packed_route_slots,
            packed_gemm_scratch_elements,
        )
        from b12x.moe.fused.w4a16.kernel import compile_w4a16_fused_moe
        from b12x.moe.fused.w4a16.prepare import (
            PreparedNF3MoeWeights,
            _make_workspace,
        )

        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        if state.t256_prep is not None:
            if runtime.t256_topk != topk:
                raise RuntimeError("Trellis256 top-k changed after preparation")
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Trellis256 weights and launch must be built before graph capture"
            )

        tail = state.num_nf3
        hidden = state.hidden_size
        inter = state.intermediate_size
        if (tail, hidden, inter) != (192, 6144, 512):
            raise RuntimeError(
                "Trellis256 admitted shape is (192, 6144, 512), "
                f"got {(tail, hidden, inter)}"
            )
        backing = state.t256_w13_backing
        rotation = state.t256_rotation
        if backing is None or rotation is None:
            raise RuntimeError("Trellis256 pre-load backing is missing")

        gate = state.tr3_slabs[("gate_proj", "trellis")]
        up = state.tr3_slabs[("up_proj", "trellis")]
        down = state.tr3_slabs[("down_proj", "trellis")]
        expected_fc1 = (tail, hidden // 16, inter // 16, 48)
        expected_fc2 = (tail, inter // 16, hidden // 16, 48)
        if (
            tuple(gate.shape) != expected_fc1
            or tuple(up.shape) != expected_fc1
            or tuple(down.shape) != expected_fc2
            or gate.dtype != torch.int16
            or up.dtype != torch.int16
            or down.dtype != torch.int16
            or not gate.is_contiguous()
            or not up.is_contiguous()
            or not down.is_contiguous()
        ):
            raise RuntimeError("Trellis256 checkpoint slab contract mismatch")
        if (
            gate.data_ptr() != backing[0].data_ptr()
            or up.data_ptr() != backing[1].data_ptr()
        ):
            raise RuntimeError("Trellis256 FC1 backing alias mismatch")

        gate_svh = state.tr3_slabs[("gate_proj", "svh")]
        up_svh = state.tr3_slabs[("up_proj", "svh")]
        down_suh = state.tr3_slabs[("down_proj", "suh")]
        rotation[:, :inter].copy_(gate_svh)
        rotation[:, inter : 2 * inter].copy_(up_svh)
        rotation[:, 2 * inter :].copy_(down_suh)
        state.t256_gate_suh = state.tr3_slabs[("gate_proj", "suh")]
        state.t256_down_svh = state.tr3_slabs[("down_proj", "svh")]
        if (
            tuple(rotation.shape) != (tail, 3 * inter)
            or tuple(state.t256_gate_suh.shape) != (tail, hidden)
            or tuple(state.t256_down_svh.shape) != (tail, hidden)
        ):
            raise RuntimeError("Trellis256 rotation-table contract mismatch")

        w13 = backing.view(torch.int32).reshape(-1)
        w2 = down.view(torch.int32).reshape(-1)
        expected_w13 = 2 * tail * (hidden // 16) * (inter // 16) * 24
        expected_w2 = tail * (inter // 16) * (hidden // 16) * 24
        if (
            not w13.is_contiguous()
            or not w2.is_contiguous()
            or w13.numel() != expected_w13
            or w2.numel() != expected_w2
        ):
            raise RuntimeError("Trellis256 fused weight view contract mismatch")

        device = w13.device
        if runtime.t256_dummy_scale is None:
            runtime.t256_dummy_scale = torch.zeros(
                4, dtype=torch.uint8, device=device
            )
        global_scale = torch.ones(tail, dtype=torch.float32, device=device)
        prep = PreparedNF3MoeWeights(
            w13=w13,
            w13_scale=runtime.t256_dummy_scale,
            w13_global_scale=global_scale,
            w2=w2,
            w2_scale=runtime.t256_dummy_scale,
            w2_global_scale=global_scale,
            workspace=_make_workspace(device),
            hidden_size=hidden,
            intermediate_size=inter,
            num_experts=tail,
            is_gated=True,
            params_dtype=torch.bfloat16,
            fc1_tile_n=_T256_TILE_CONFIG[1],
            fc2_tile_n=_T256_TILE_CONFIG[3],
            source_format="trellis3_t256",
            w13_layout="trellis3_t256_proj",
            weight_layout="trellis3_t256",
            scale_format="e4m3_k32",
        )
        emap = torch.full(
            (state.num_experts,), -1, dtype=torch.int32, device=device
        )
        for global_id, (tier, local_id) in state.remap.items():
            if tier == 1:
                emap[global_id] = local_id
        state.t256_prep = prep
        state.t256_emap = emap

        route_capacity = _T256_MAX_M * topk
        if runtime.t256_launch is None:
            props = torch.cuda.get_device_properties(device)
            sms = int(props.multi_processor_count)
            max_shared_mem = int(
                getattr(props, "shared_memory_per_block_optin", 101_376)
            )
            route_slots = max_packed_route_slots(route_capacity, 64, tail)
            runtime.t256_launch = compile_w4a16_fused_moe(
                size_m=route_capacity,
                hidden_size=hidden,
                intermediate_size=inter,
                num_experts=tail,
                top_k=1,
                activation="silu",
                apply_router_weight_on_input=False,
                zero_fc2_output=True,
                moe_block_size=64,
                max_m_blocks=(route_slots + 63) // 64,
                element_dtype="bf16",
                fast_math=True,
                sms=sms,
                max_shared_mem=max_shared_mem,
                weight_layout="trellis3_t256",
                scale_format="e4m3_k32",
                w13_layout="trellis3_t256_proj",
                force_tile_config=_T256_TILE_CONFIG,
                intermediate_rotation=True,
            )
            buffers = make_w4a16_packed_buffers(
                prep,
                m=route_capacity,
                topk=1,
                dtype=torch.bfloat16,
                device=device,
                route_num_experts=tail,
            )
            need_blocks = (route_slots + 63) // 64
            need_fc1 = packed_gemm_scratch_elements(
                size_n=2 * inter,
                route_slots=route_slots,
                moe_block_size=64,
                sms=sms,
            )
            need_fc2 = packed_gemm_scratch_elements(
                size_n=hidden,
                route_slots=route_slots,
                moe_block_size=64,
                sms=sms,
            )
            updates = {}
            if buffers.packed_route_indices.numel() < route_slots:
                updates["packed_route_indices"] = torch.empty(
                    route_slots, dtype=torch.int32, device=device
                )
            if buffers.block_expert_ids.numel() < need_blocks:
                updates["block_expert_ids"] = torch.empty(
                    need_blocks, dtype=torch.int32, device=device
                )
            if buffers.fc1_c_tmp is None or buffers.fc1_c_tmp.numel() < need_fc1:
                updates["fc1_c_tmp"] = torch.empty(
                    need_fc1, dtype=torch.float32, device=device
                )
            if buffers.fc2_c_tmp is None or buffers.fc2_c_tmp.numel() < need_fc2:
                updates["fc2_c_tmp"] = torch.empty(
                    need_fc2, dtype=torch.float32, device=device
                )
            if updates:
                buffers = dataclasses.replace(buffers, **updates)
            runtime.t256_buffers = buffers
            runtime.t256_route_capacity = route_capacity
            runtime.t256_topk = topk
            runtime.t256_ones = torch.ones(
                (route_capacity, 1), dtype=torch.float32, device=device
            )
            runtime.t256_route_token = torch.arange(
                _T256_MAX_M, dtype=torch.int64, device=device
            ).repeat_interleave(topk)
            runtime.t256_identity_emap = torch.arange(
                tail, dtype=torch.int32, device=device
            )
            logger.info(
                "Trellis256 runtime ready: m=%d..%d routes=%d tiles=%s",
                _T256_MIN_M,
                _T256_MAX_M,
                route_capacity,
                _T256_TILE_CONFIG,
            )
        elif (
            runtime.t256_route_capacity != route_capacity
            or runtime.t256_topk != topk
        ):
            raise RuntimeError("Trellis256 shared runtime geometry changed")

        runtime.t256_prepared_layers.add(layer.layer_name)
        if (
            not runtime.t256_cache_released
            and len(runtime.t256_prepared_layers)
            == len(self.quant_config.hybrid_bit_map)
        ):
            before = torch.cuda.memory_reserved(device)
            gc.collect()
            torch.cuda.empty_cache()
            runtime.t256_cache_released = True
            logger.info(
                "Trellis256 released eager setup cache: %d -> %d MiB reserved",
                before >> 20,
                torch.cuda.memory_reserved(device) >> 20,
            )

    def _prepare_grid188(self, layer: RoutedExperts, topk: int) -> None:
        if _PLANNED_TAIL_ENABLED and layer.hybrid_state.num_nf3:
            self._build_planned_trellis_moe(layer, topk)
            if _MIXED_TRELLIS_ENABLED:
                self._build_mixed_trellis_moe(layer, topk)
        elif _T256_ENABLED and layer.hybrid_state.num_nf3:
            self._build_trellis256_moe(layer, topk)

    def _run_trellis256(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from b12x.moe.fused.tr3 import hadamard_128
        from b12x.moe.fused.w4a16.kernel import run_w4a16_moe

        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        prep = state.t256_prep
        buffers = runtime.t256_buffers
        if prep is None or buffers is None or runtime.t256_launch is None:
            raise RuntimeError("Trellis256 runtime was not prepared before dispatch")

        m = int(x.shape[0])
        topk = int(topk_ids.shape[1])
        routes = m * topk
        if (
            topk != runtime.t256_topk
            or routes > int(runtime.t256_route_capacity)
        ):
            raise RuntimeError(
                f"Trellis256 route capacity mismatch: m/topk/routes={m}/{topk}/{routes}"
            )
        key = (m, topk, routes)
        if key not in runtime.t256_seen_m:
            runtime.t256_seen_m.add(key)
            logger.info("Trellis256 dispatch: m=%d topk=%d routes=%d", *key)

        emap = state.t256_emap
        assert emap is not None
        global_ids = topk_ids.to(torch.int64)
        in_range = (global_ids >= 0) & (global_ids < int(emap.numel()))
        safe_ids = global_ids.clamp(min=0, max=int(emap.numel()) - 1)
        local_ids = emap[safe_ids].reshape(-1)
        valid = in_range.reshape(-1) & (local_ids >= 0)
        gather_ids = local_ids.clamp(min=0).to(torch.int64)
        route_token = runtime.t256_route_token[:routes]

        x_half = x.index_select(0, route_token).to(torch.float16)
        x_half.mul_(state.t256_gate_suh.index_select(0, gather_ids))
        x_rotated = torch.empty_like(x_half)
        hadamard_128(x_half, x_rotated)
        x_bf16 = x_rotated.to(torch.bfloat16)
        rotation = state.t256_rotation.index_select(0, gather_ids).contiguous()
        ids1 = local_ids.clamp(min=0).reshape(routes, 1).to(torch.int32).contiguous()
        fc2 = run_w4a16_moe(
            x_bf16,
            prep,
            runtime.t256_ones[:routes],
            ids1,
            activation="silu",
            intermediate_cache13=buffers.intermediate_cache13,
            intermediate_cache2=buffers.intermediate_cache2,
            output=buffers.output[:routes],
            fc1_c_tmp=buffers.fc1_c_tmp,
            fc2_c_tmp=buffers.fc2_c_tmp,
            packed_route_indices=buffers.packed_route_indices,
            block_expert_ids=buffers.block_expert_ids,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_map=runtime.t256_identity_emap,
            fused_launch=runtime.t256_launch,
            intermediate_rotation_scales=rotation,
        )
        fc2_half = fc2.to(torch.float16)
        output_rotated = torch.empty_like(fc2_half)
        hadamard_128(fc2_half, output_rotated)
        output_rotated.mul_(state.t256_down_svh.index_select(0, gather_ids))
        route_weights = (
            topk_weights.reshape(routes, 1).float()
            * valid.reshape(routes, 1).float()
        )
        return (
            output_rotated.float()
            .mul_(route_weights)
            .view(m, topk, state.hidden_size)
            .sum(1)
        )

    def _ensure_tr3_runtime(self, layer: RoutedExperts, m: int, topk: int) -> None:
        from b12x.moe.fused.tr3 import max_concurrency

        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        if runtime.tr3_x is not None:
            if runtime.tr3_topk != topk or runtime.tr3_num_experts != state.num_nf3:
                raise RuntimeError("TR3 routing geometry changed after runtime setup")
            state.tr3_runtime_ready = True
            return
        if runtime.tr3_cap <= 0:
            raise ValueError("B12X_TR3_MAX_TOKENS_PER_EXPERT must be positive")

        max_m = max(int(self.moe.max_num_tokens), int(m))
        device = state.tr3_slabs[("gate_proj", "trellis")].device
        concurrency = max_concurrency(device)
        cap = runtime.tr3_cap
        hidden, inter = state.hidden_size, state.intermediate_size
        runtime.tr3_max_m = max_m
        runtime.tr3_topk = topk
        runtime.tr3_num_experts = state.num_nf3
        runtime.tr3_x = torch.empty((max_m, hidden), dtype=torch.float16, device=device)
        runtime.tr3_output = torch.empty(
            (max_m, hidden), dtype=torch.float32, device=device
        )
        runtime.tr3_temp_gate = torch.empty(
            (concurrency, cap, hidden), dtype=torch.float16, device=device
        )
        runtime.tr3_temp_up = torch.empty_like(runtime.tr3_temp_gate)
        runtime.tr3_temp_intermediate_gate = torch.empty(
            (concurrency, cap, inter), dtype=torch.float16, device=device
        )
        runtime.tr3_temp_intermediate_up = torch.empty_like(
            runtime.tr3_temp_intermediate_gate
        )
        runtime.tr3_flat_token = torch.arange(
            cap, dtype=torch.int64, device=device
        ).repeat_interleave(topk)
        runtime.tr3_ones = torch.ones(cap * topk, dtype=torch.int64, device=device)
        runtime.tr3_expert_count = torch.empty(
            state.num_nf3 + 1, dtype=torch.int64, device=device
        )
        runtime.tr3_expert_offsets = torch.empty_like(runtime.tr3_expert_count)
        runtime.tr3_token_sorted = torch.empty(
            max_m * topk, dtype=torch.int64, device=device
        )
        runtime.tr3_weight_sorted = torch.empty(
            max_m * topk, dtype=torch.float16, device=device
        )
        state.tr3_runtime_ready = True
        logger.info(
            "B12X TR3 runtime ready: max_m=%d cap=%d concurrency=%d",
            max_m,
            cap,
            concurrency,
        )

    def _run_tr3(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from b12x.moe.fused.tr3 import run

        state: _HybridLayerState = layer.hybrid_state
        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        m = int(x.shape[0])
        topk = int(topk_ids.shape[1])
        x_half = runtime.tr3_x[:m]
        x_half.copy_(x)
        output = runtime.tr3_output[:m]
        output.zero_()
        cap = runtime.tr3_cap

        local_ids = state.tr3_emap[topk_ids.long()]
        weights_half = topk_weights.to(torch.float16)
        for start in range(0, m, cap):
            count = min(cap, m - start)
            flat_ids = local_ids[start : start + count].reshape(-1)
            order = torch.argsort(flat_ids)
            tokens = runtime.tr3_flat_token[: count * topk].index_select(0, order)
            weights = (
                weights_half[start : start + count]
                .reshape(-1)
                .index_select(0, order)
                .contiguous()
            )
            expert_count = runtime.tr3_expert_count
            expert_count.zero_()
            expert_count.scatter_add_(0, flat_ids, runtime.tr3_ones[: count * topk])
            run(
                x_half[start : start + count],
                output[start : start + count],
                expert_count,
                tokens.contiguous(),
                weights,
                runtime.tr3_temp_gate,
                runtime.tr3_temp_up,
                runtime.tr3_temp_intermediate_gate,
                runtime.tr3_temp_intermediate_up,
                state.tr3_pointer_tables,
                active_experts=min(count * topk, state.num_nf3),
            )
        return output

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        state: _HybridLayerState = layer.hybrid_state
        if state.num_nf3 == 0:
            return super().apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )

        runtime: _Tr3SharedRuntime = self.quant_config.shared_runtime
        m = int(x.shape[0])
        topk = int(topk_ids.shape[1])
        if not state.runtime_ready:
            self._ensure_runtime(layer, m, topk)
        use_decode_plan = (
            _PLANNED_TAIL_ENABLED
            and _PLANNED_TAIL_MIN_M <= m <= _PLANNED_TAIL_MAX_M
        )
        use_prefill_plan = (
            _PLANNED_TAIL_ENABLED
            and _PLANNED_PREFILL_ENABLED
            and _PLANNED_TAIL_MAX_M < m <= _PLANNED_PREFILL_MAX_M
        )
        use_planned = use_decode_plan or use_prefill_plan
        if not use_planned and not state.tr3_runtime_ready:
            self._ensure_tr3_runtime(layer, m, topk)
        if m > int(runtime.max_m) or (
            not use_planned and m > int(runtime.tr3_max_m)
        ):
            raise RuntimeError(f"TR3 batch {m} exceeds configured runtime capacity")
        weights = topk_weights.float()
        if not weights.is_contiguous():
            weights = weights.contiguous()
        use_mixed = (
            _MIXED_TRELLIS_ENABLED
            and 1 <= m <= _MIXED_TRELLIS_MAX_M
        )
        if use_mixed:
            out_mixed = self._run_mixed_trellis(
                layer, x, weights, topk_ids
            )
            if (
                _MIXED_TRELLIS_PARITY
                and not state.mixed_parity_checked
                and not torch.cuda.is_current_stream_capturing()
            ):
                reference_kept = self._run_kept(
                    layer, x, weights, topk_ids, m <= 8
                )
                reference_tail = self._run_planned_trellis(
                    layer, x, weights, topk_ids
                )
                reference_tail.add_(reference_kept)
                actual32 = out_mixed.float()
                reference32 = reference_tail.float()
                actual_norm = actual32.norm()
                reference_norm = reference32.norm()
                relative_error = (
                    (actual32 - reference32).norm()
                    / reference_norm.clamp_min(1.0e-9)
                )
                rel = float(relative_error.item())
                if (
                    float(actual_norm.item()) <= 1.0e-9
                    and float(reference_norm.item()) <= 1.0e-9
                ):
                    cos = 1.0
                else:
                    cosine = torch.nn.functional.cosine_similarity(
                        actual32.flatten(), reference32.flatten(), dim=0
                    )
                    cos = float(cosine.item())
                logger.info(
                    "Mixed Trellis parity %s m=%d: "
                    "relative_error=%.8f cosine=%.8f",
                    layer.layer_name,
                    m,
                    rel,
                    cos,
                )
                if (
                    rel >= _MIXED_TRELLIS_PARITY_REL_MAX
                    or cos <= _MIXED_TRELLIS_PARITY_COS_MIN
                ):
                    raise RuntimeError(
                        "mixed Trellis parity failed for "
                        f"{layer.layer_name}: relative_error={rel}, cosine={cos}"
                    )
                state.mixed_parity_checked = True
            return out_mixed
        use_overlap = use_planned and _PLANNED_TAIL_OVERLAP
        if use_overlap:
            if (
                runtime.planned_stream is None
                or state.planned_fork_event is None
                or state.planned_done_event is None
            ):
                raise RuntimeError("planned Trellis overlap was not prepared")
            current_stream = torch.cuda.current_stream(x.device)
            state.planned_fork_event.record(current_stream)
            runtime.planned_stream.wait_event(state.planned_fork_event)
            with torch.cuda.stream(runtime.planned_stream):
                out_tail = self._run_planned_trellis(
                    layer, x, weights, topk_ids
                )
                state.planned_done_event.record(runtime.planned_stream)
            out_kept = self._run_kept(layer, x, weights, topk_ids, m <= 8)
            current_stream.wait_event(state.planned_done_event)
        else:
            out_kept = self._run_kept(layer, x, weights, topk_ids, m <= 8)
        if use_planned:
            if not use_overlap:
                out_tail = self._run_planned_trellis(
                    layer, x, weights, topk_ids
                )
            parity_pending = (
                not state.planned_prefill_parity_checked
                if use_prefill_plan
                else not state.planned_parity_checked
            )
            if (
                _PLANNED_TAIL_PARITY
                and parity_pending
                and (
                    not use_prefill_plan
                    or m <= _PLANNED_PREFILL_PARITY_MAX_M
                )
                and not torch.cuda.is_current_stream_capturing()
            ):
                if not state.tr3_runtime_ready:
                    self._ensure_tr3_runtime(layer, m, topk)
                reference = self._run_tr3(layer, x, weights, topk_ids)
                actual32 = out_tail.float()
                reference32 = reference.float()
                relative_error = (
                    (actual32 - reference32).norm()
                    / reference32.norm().clamp_min(1.0e-9)
                )
                cosine = torch.nn.functional.cosine_similarity(
                    actual32.flatten(), reference32.flatten(), dim=0
                )
                rel = float(relative_error.item())
                cos = float(cosine.item())
                logger.info(
                    "Planned Trellis parity %s m=%d: relative_error=%.8f cosine=%.8f",
                    layer.layer_name, m, rel, cos
                )
                rel_max = (
                    _PLANNED_PREFILL_PARITY_REL_MAX
                    if use_prefill_plan
                    else _PLANNED_TAIL_PARITY_REL_MAX
                )
                cos_min = (
                    _PLANNED_PREFILL_PARITY_COS_MIN
                    if use_prefill_plan
                    else _PLANNED_TAIL_PARITY_COS_MIN
                )
                if (
                    rel >= rel_max
                    or cos <= cos_min
                ):
                    raise RuntimeError(
                        "planned Trellis parity failed for "
                        f"{layer.layer_name}: relative_error={rel}, cosine={cos}"
                    )
                if use_prefill_plan:
                    state.planned_prefill_parity_checked = True
                else:
                    state.planned_parity_checked = True
        elif _T256_ENABLED and _T256_MIN_M <= m <= _T256_MAX_M:
            out_tail = self._run_trellis256(
                layer, x, weights, topk_ids
            )
        else:
            out_tail = self._run_tr3(layer, x, weights, topk_ids)
        out_tail.add_(out_kept)
        return out_tail.to(out_kept.dtype)


NvFp4Tr3HybridConfig.FusedMoEMethodCls = NvFp4Tr3HybridMoEMethod
