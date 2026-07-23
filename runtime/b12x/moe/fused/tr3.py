"""TR3 trellis fused-MoE runtime for B12X.

The checkpoint stores each TP-local expert projection as an ExLlamaV3 trellis
plus the two Hadamard vectors used by the calibrated LDLQ transform.  This
module owns the serving ABI used by vLLM so the quantizer is independent of the
extension package layout.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
import sys
from typing import Any

import torch


_REQUIRED_OPS = (
    "exl3_moe",
    "exl3_moe_max_concurrency",
    "had_r_128",
)


@lru_cache(maxsize=1)
def _extension() -> Any:
    """Load the TR3 CUDA extension after Torch and its ABI shim."""
    import torch  # noqa: F401 -- libtorch must be resident before the extension

    candidates = (
        os.getenv("B12X_TR3_TORCH_ABI_SHIM", ""),
        str(Path(__file__).with_name("libb12x_tr3_torch_compat.so")),
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            break

    try:
        import exllamav3_ext as extension
    except ImportError as import_error:
        extension_path = next(
            Path(__file__).parent.glob("exllamav3_ext*.so"), None
        )
        if extension_path is None:
            raise RuntimeError(
                "The B12X TR3 backend requires the bundled exllamav3_ext CUDA "
                "extension. Rebuild the image with TR3 support enabled."
            ) from import_error
        spec = importlib.util.spec_from_file_location("exllamav3_ext", extension_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load TR3 extension {extension_path}")
        extension = importlib.util.module_from_spec(spec)
        sys.modules["exllamav3_ext"] = extension
        spec.loader.exec_module(extension)

    missing = [name for name in _REQUIRED_OPS if not hasattr(extension, name)]
    if missing:
        raise RuntimeError(
            "The installed TR3 extension lacks required operations: "
            + ", ".join(missing)
        )
    return extension


def hadamard_128(input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> None:
    """Apply ExLlamaV3's block-128 Hadamard rotation."""
    if input_tensor.dtype != torch.float16 or output_tensor.dtype != torch.float16:
        raise TypeError("TR3 Hadamard input and output must be float16")
    if input_tensor.shape != output_tensor.shape:
        raise ValueError("TR3 Hadamard input and output shapes must match")
    _extension().had_r_128(input_tensor, output_tensor, None, None, 1.0)


def max_concurrency(device: torch.device | int | None = None) -> int:
    """Return the maximum persistent expert groups supported by a device."""
    if device is None:
        device_index = torch.cuda.current_device()
    elif isinstance(device, int):
        device_index = device
    else:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
    return int(_extension().exl3_moe_max_concurrency(device_index))


def run(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    expert_count: torch.Tensor,
    token_sorted: torch.Tensor,
    weight_sorted: torch.Tensor,
    temp_gate: torch.Tensor,
    temp_up: torch.Tensor,
    temp_intermediate_gate: torch.Tensor,
    temp_intermediate_up: torch.Tensor,
    pointer_tables: tuple[torch.Tensor, ...],
    *,
    bits: int = 3,
    activation_limit: float = 0.0,
    active_experts: int | None = None,
) -> None:
    """Run a pre-grouped TR3 expert MLP."""
    if len(pointer_tables) != 9:
        raise ValueError(f"TR3 requires nine pointer tables, got {len(pointer_tables)}")
    _extension().exl3_moe(
        hidden_states,
        output,
        expert_count,
        token_sorted,
        weight_sorted,
        temp_gate,
        temp_up,
        temp_intermediate_gate,
        temp_intermediate_up,
        0,
        bits,
        bits,
        bits,
        *pointer_tables,
        True,
        False,
        True,
        False,
        True,
        False,
        activation_limit,
        active_experts if active_experts is not None else expert_count.numel() - 1,
    )


