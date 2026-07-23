#!/opt/venv/bin/python
"""Enable the stock extension's fused large-batch TR3 route on v19."""

import os
from pathlib import Path

TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "quantization/nvfp4_tr3_hybrid.py"
)
WRAPPER_TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/b12x/moe/fused/tr3.py"
)
ENTRYPOINT_TARGET = Path("/usr/local/bin/serve-glm52-v16.sh")
MTP_TARGET = Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/"
    "glm4_moe_lite_mtp.py"
)

source = TARGET.read_text()
old_import = "        from b12x.moe.fused.tr3 import run\n"
new_import = "        from b12x.moe.fused.tr3 import _extension, run\n"
anchor = """        cap = runtime.tr3_cap

        local_ids = state.tr3_emap[topk_ids.long()]
"""
fused_path = """        cap = runtime.tr3_cap
        if m > cap:
            extension = _extension()
            if not hasattr(extension, "exl3_moe_fused"):
                raise RuntimeError("fused TR3 extension lacks exl3_moe_fused")
            extension.exl3_moe_fused(
                x_half,
                output,
                topk_ids,
                topk_weights,
                state.tr3_emap,
                runtime.tr3_expert_count,
                runtime.tr3_expert_offsets,
                runtime.tr3_token_sorted,
                runtime.tr3_weight_sorted,
                runtime.tr3_temp_gate,
                runtime.tr3_temp_up,
                runtime.tr3_temp_intermediate_gate,
                runtime.tr3_temp_intermediate_up,
                0,
                3,
                3,
                3,
                *state.tr3_pointer_tables,
                True,
                False,
                True,
                False,
                True,
                False,
                0.0,
                0,
            )
            return output

        local_ids = state.tr3_emap[topk_ids.long()]
"""

if source.count(old_import) != 1:
    raise RuntimeError("unexpected v19 TR3 import anchor")
if source.count(anchor) != 1:
    raise RuntimeError("unexpected v19 TR3 large-batch anchor")
tile_config = os.getenv("B12X_TRELLIS256_TILE_CONFIG", "")
if tile_config:
    values = tuple(int(value) for value in tile_config.split(","))
    valid_pairs = {(64, 256), (64, 128), (128, 64)}
    if (
        len(values) != 4
        or values[:2] not in valid_pairs
        or values[2:] not in valid_pairs
    ):
        raise ValueError(
            "B12X_TRELLIS256_TILE_CONFIG must contain two valid K,N pairs; "
            "valid pairs are 64,256, 64,128, and 128,64"
        )
    tile_anchor = "_T256_TILE_CONFIG = (64, 256, 64, 256)"
    if source.count(tile_anchor) != 1:
        raise RuntimeError("unexpected v19 Trellis256 tile-config anchor")
    source = source.replace(tile_anchor, f"_T256_TILE_CONFIG = {values!r}")

source = source.replace(old_import, new_import).replace(anchor, fused_path)
TARGET.write_text(source)

mtp_source = MTP_TARGET.read_text()
predictor_logits = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        logits = self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )
        return logits
"""
predictor_local_argmax = predictor_logits + """
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        \"\"\"Return the vocab-parallel argmax without gathering full logits.\"\"\"
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return self.logits_processor.get_top_tokens(
            mtp_layer.shared_head.head,
            mtp_layer.shared_head(hidden_states),
        )
"""
model_logits = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)
"""
model_local_argmax = model_logits + """
    def get_top_tokens(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.get_top_tokens(hidden_states, spec_step_idx)
"""
if mtp_source.count(predictor_logits) != 1 or mtp_source.count(model_logits) != 1:
    raise RuntimeError("unexpected GLM MTP logits anchors")
mtp_source = mtp_source.replace(
    predictor_logits, predictor_local_argmax
).replace(model_logits, model_local_argmax)
MTP_TARGET.write_text(mtp_source)
print(f"Patched local-argmax support in {MTP_TARGET}", flush=True)

wrapper = WRAPPER_TARGET.read_text()
active_experts_arg = """        activation_limit,
        active_experts if active_experts is not None else expert_count.numel() - 1,
"""
legacy_abi_arg = """        activation_limit,
"""
if wrapper.count(active_experts_arg) != 1:
    raise RuntimeError("unexpected v19 exl3_moe ABI anchor")
WRAPPER_TARGET.write_text(wrapper.replace(active_experts_arg, legacy_abi_arg))
print(f"Patched legacy exl3_moe ABI in {WRAPPER_TARGET}", flush=True)

entrypoint = ENTRYPOINT_TARGET.read_text()
command_tail = '''  "${spec_arg[@]}")'''
forwarding_tail = '''  "${spec_arg[@]}" "$@")'''
if entrypoint.count(command_tail) != 1:
    raise RuntimeError("unexpected v19 entrypoint command anchor")
entrypoint = entrypoint.replace(command_tail, forwarding_tail)
if os.getenv("VLLM_MTP_LOCAL_ARGMAX", "0") == "1":
    spec_tail = '"draft_sample_method":"probabilistic"}'
    local_argmax_tail = (
        '"draft_sample_method":"probabilistic",'
        '"use_local_argmax_reduction":true}'
    )
    if entrypoint.count(spec_tail) != 1:
        raise RuntimeError("unexpected MTP speculative-config anchor")
    entrypoint = entrypoint.replace(spec_tail, local_argmax_tail)
ENTRYPOINT_TARGET.write_text(entrypoint)
print(f"Patched argument forwarding in {ENTRYPOINT_TARGET}", flush=True)

print(f"Patched fused large-batch TR3 path in {TARGET}", flush=True)
