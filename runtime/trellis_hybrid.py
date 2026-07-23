"""Persistent mixed NVFP4/Trellis256 MoE decode kernel.

This specialization routes the checkpoint's 64 NVFP4 and 192 rank-sliced
Trellis experts inside one persistent grid.  It keeps one route-major FC1/FC2
workspace, applies the Trellis rotations only to tail routes, and performs one
mixed fp32 top-k reduction at the end of the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cutlass_dsl import Int32

from sparkinfer._lib.compiler import KernelCompileSpec, compile as sparkinfer_compile
from sparkinfer._lib.runtime_control import raise_if_kernel_resolution_frozen
from sparkinfer._lib.utils import current_cuda_stream, make_ptr

from . import kernel as _base


@dataclass(frozen=True)
class W4A16TrellisHybridCompileResult:
    compiled: object
    size_m: int
    hidden_size: int
    intermediate_size: int
    top_k: int
    activation: str
    element_dtype: str
    fast_math: bool
    map_slots: int
    tier0_num_experts: int
    tier1_num_experts: int
    fc1_tile_n: int
    fc1_tile_k: int
    fc2_tile_n: int
    fc2_tile_k: int
    moe_block_size: int
    max_m_blocks: int
    cta_threads: int
    blocks_per_sm: int
    shared_memory_bytes: int
    schedule_whole_tiles: bool
    direct_topk_routes: bool
    tc_decode_fused_sum: bool
    registers_per_thread: int
    local_memory_bytes: int
    trellis_bits: int


class W4A16TrellisHybridKernel:
    """One-grid E64 NVFP4 + E192 full-rotation Trellis decode kernel."""

    ABI_VERSION = 1

    def __init__(
        self,
        *,
        tier0: _base.W4A16FusedMoeKernel,
        tier1: _base.W4A16FusedMoeKernel,
        map_slots: int,
    ) -> None:
        if tier0.weight_layout != "packed":
            raise ValueError("mixed Trellis tier0 must use packed NVFP4")
        if tier1.weight_layout != "trellis3_t256":
            raise ValueError("mixed Trellis tier1 must use trellis3_t256")
        if tier0.intermediate_rotation or tier0.dual_a or tier0.route_major_a:
            raise ValueError("mixed Trellis tier0 must be an ordinary W4A16 tier")
        if not tier1.intermediate_rotation or not tier1.dual_a or not tier1.route_major_a:
            raise ValueError("mixed Trellis tier1 requires route-major dual-A rotation")
        for name, moe in (("tier0", tier0), ("tier1", tier1)):
            if not moe.direct_topk_routes:
                raise ValueError(f"mixed Trellis {name} requires direct top-k routes")
            if moe.tc_decode_fused_sum:
                raise ValueError(f"mixed Trellis {name} defers the top-k reduction")
            if not moe.defer_router_weight:
                raise ValueError(f"mixed Trellis {name} must defer router weights")
            if not moe.activation_is_gated:
                raise ValueError(f"mixed Trellis {name} requires gated SiLU")
            if moe.collect_activation_amax or moe.zero_fc2_output:
                raise ValueError(f"mixed Trellis {name} has incompatible epilogues")
        for attr in (
            "size_m",
            "hidden_size",
            "intermediate_size",
            "fc1_cols",
            "top_k",
            "moe_block_size",
            "activation",
            "element_dtype",
            "is_fp16",
            "fast_math",
            "cta_threads",
            "sms",
            "barrier_count_off",
            "barrier_sense_off",
        ):
            if getattr(tier0, attr) != getattr(tier1, attr):
                raise ValueError(f"mixed Trellis tiers disagree on {attr}")
        for phase in ("fc1", "fc2"):
            gemm0 = getattr(tier0, phase)
            gemm1 = getattr(tier1, phase)
            if (gemm0.n_tiles, gemm0.k_tiles, gemm0.tile_n, gemm0.tile_k) != (
                gemm1.n_tiles,
                gemm1.k_tiles,
                gemm1.tile_n,
                gemm1.tile_k,
            ):
                raise ValueError(f"mixed Trellis tiers disagree on {phase} tiling")
        if tier0.element_dtype != "bf16":
            raise ValueError("mixed Trellis v1 requires bf16 serving activations")
        if tier0.activation != "silu":
            raise ValueError("mixed Trellis v1 requires plain gated SiLU")
        if tier0.hidden_size % 128 or tier0.intermediate_size % 128:
            raise ValueError("mixed Trellis rotations require H and I divisible by 128")
        if int(map_slots) < tier0.num_experts + tier1.num_experts:
            raise ValueError("mixed Trellis descriptor map is too small")
        if tier0.num_experts > 256 or tier1.num_experts > 256:
            raise ValueError("mixed Trellis local expert ids must fit eight bits")

        self.tier0 = tier0
        self.tier1 = tier1
        self.map_slots = int(map_slots)
        self.size_m = tier0.size_m
        self.hidden_size = tier0.hidden_size
        self.intermediate_size = tier0.intermediate_size
        self.top_k = tier0.top_k
        self.element_dtype = tier0.element_dtype
        self.cta_threads = tier0.cta_threads
        self.sms = tier0.sms
        self.blocks_per_sm = min(tier0.blocks_per_sm, tier1.blocks_per_sm)
        self.shared_words = max(tier0.shared_words, tier1.shared_words)

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            "w4a16_trellis_hybrid",
            self.ABI_VERSION,
            self.map_slots,
            self.tier0.__cache_key__,
            self.tier1.__cache_key__,
            self.shared_words,
        )

    @cute.jit
    def _emit_tile(
        self,
        is_fc1: cutlass.Constexpr,
        a_input_flat: cute.Tensor,
        a_gate_flat: cute.Tensor,
        a_up_flat: cute.Tensor,
        t0_b_i32_flat: cute.Tensor,
        t0_scales_i32_flat: cute.Tensor,
        t0_global_scale: cute.Tensor,
        t1_b_i32_flat: cute.Tensor,
        t1_scales_i32_flat: cute.Tensor,
        t1_global_scale: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        global_topk_ids_i32_flat: cute.Tensor,
        tier_local_map_i32_flat: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        active_size_m: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
    ):
        gid = global_topk_ids_i32_flat[route_block_idx].to(Int32)
        if gid >= Int32(0) and gid < Int32(self.map_slots):
            descriptor = tier_local_map_i32_flat[gid].to(Int32)
            if descriptor >= Int32(0):
                tier = descriptor >> Int32(8)
                local_expert = descriptor & Int32(0xFF)
                if tier == Int32(0):
                    if local_expert < Int32(self.tier0.num_experts):
                        gemm = self.tier0.fc1 if cutlass.const_expr(is_fc1) else self.tier0.fc2
                        gemm._run_tile(
                            a_input_flat,
                            a_input_flat,
                            t0_b_i32_flat,
                            c_bf16_flat,
                            t0_scales_i32_flat,
                            t0_global_scale,
                            global_topk_ids_i32_flat,
                            topk_weights_flat,
                            c_tmp_f32_flat,
                            locks_i32_flat,
                            smem_base,
                            tid,
                            route_block_idx,
                            local_expert,
                            output_n_tile,
                            reduce_k_tile,
                            reduce_tile_count,
                            reduce_slice_count,
                            reduce_slice_idx,
                            lock_slot,
                            active_size_m,
                        )
                elif tier == Int32(1):
                    if local_expert < Int32(self.tier1.num_experts):
                        gemm = self.tier1.fc1 if cutlass.const_expr(is_fc1) else self.tier1.fc2
                        gemm._run_tile(
                            a_gate_flat if cutlass.const_expr(is_fc1) else a_input_flat,
                            a_up_flat if cutlass.const_expr(is_fc1) else a_input_flat,
                            t1_b_i32_flat,
                            c_bf16_flat,
                            t1_scales_i32_flat,
                            t1_global_scale,
                            global_topk_ids_i32_flat,
                            topk_weights_flat,
                            c_tmp_f32_flat,
                            locks_i32_flat,
                            smem_base,
                            tid,
                            route_block_idx,
                            local_expert,
                            output_n_tile,
                            reduce_k_tile,
                            reduce_tile_count,
                            reduce_slice_count,
                            reduce_slice_idx,
                            lock_slot,
                            active_size_m,
                        )

    @cute.jit
    def _rotate_tail_inputs(
        self,
        x_flat: cute.Tensor,
        a_gate_flat: cute.Tensor,
        a_up_flat: cute.Tensor,
        gate_suh_flat: cute.Tensor,
        up_suh_flat: cute.Tensor,
        global_ids: cute.Tensor,
        tier_map: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        hblocks = Int32(self.hidden_size // 128)
        routes = active_m * Int32(self.top_k)
        unit = cta * warps_per_cta + warp
        stride = grid_x * warps_per_cta
        elem = lane * Int32(4)
        while unit < routes * hblocks:
            route = unit // hblocks
            block = unit - route * hblocks
            gid = global_ids[route].to(Int32)
            if gid >= Int32(0) and gid < Int32(self.map_slots):
                descriptor = tier_map[gid].to(Int32)
                if descriptor >= Int32(256):
                    expert = descriptor & Int32(0xFF)
                    if expert < Int32(self.tier1.num_experts):
                        col = block * Int32(128) + elem
                        token = route // Int32(self.top_k)
                        xbase = token * Int32(self.hidden_size) + col
                        sbase = expert * Int32(self.hidden_size) + col
                        obase = route * Int32(self.hidden_size) + col
                        x0 = x_flat[xbase + Int32(0)].to(cutlass.Float32)
                        x1 = x_flat[xbase + Int32(1)].to(cutlass.Float32)
                        x2 = x_flat[xbase + Int32(2)].to(cutlass.Float32)
                        x3 = x_flat[xbase + Int32(3)].to(cutlass.Float32)
                        g0 = cutlass.Float16(x0 * gate_suh_flat[sbase + Int32(0)].to(cutlass.Float32)).to(cutlass.Float32)
                        g1 = cutlass.Float16(x1 * gate_suh_flat[sbase + Int32(1)].to(cutlass.Float32)).to(cutlass.Float32)
                        g2 = cutlass.Float16(x2 * gate_suh_flat[sbase + Int32(2)].to(cutlass.Float32)).to(cutlass.Float32)
                        g3 = cutlass.Float16(x3 * gate_suh_flat[sbase + Int32(3)].to(cutlass.Float32)).to(cutlass.Float32)
                        u0 = cutlass.Float16(x0 * up_suh_flat[sbase + Int32(0)].to(cutlass.Float32)).to(cutlass.Float32)
                        u1 = cutlass.Float16(x1 * up_suh_flat[sbase + Int32(1)].to(cutlass.Float32)).to(cutlass.Float32)
                        u2 = cutlass.Float16(x2 * up_suh_flat[sbase + Int32(2)].to(cutlass.Float32)).to(cutlass.Float32)
                        u3 = cutlass.Float16(x3 * up_suh_flat[sbase + Int32(3)].to(cutlass.Float32)).to(cutlass.Float32)
                        gh = self.tier1._had128_quad(g0, g1, g2, g3, lane)
                        uh = self.tier1._had128_quad(u0, u1, u2, u3, lane)
                        a_gate_flat[obase + Int32(0)] = self.tier1._cast_elem(gh[0])
                        a_gate_flat[obase + Int32(1)] = self.tier1._cast_elem(gh[1])
                        a_gate_flat[obase + Int32(2)] = self.tier1._cast_elem(gh[2])
                        a_gate_flat[obase + Int32(3)] = self.tier1._cast_elem(gh[3])
                        a_up_flat[obase + Int32(0)] = self.tier1._cast_elem(uh[0])
                        a_up_flat[obase + Int32(1)] = self.tier1._cast_elem(uh[1])
                        a_up_flat[obase + Int32(2)] = self.tier1._cast_elem(uh[2])
                        a_up_flat[obase + Int32(3)] = self.tier1._cast_elem(uh[3])
            unit += stride

    @cute.jit
    def _activate_routes(
        self,
        fc1_flat: cute.Tensor,
        activated_flat: cute.Tensor,
        rotations_flat: cute.Tensor,
        global_ids: cute.Tensor,
        tier_map: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        iblocks = Int32(self.intermediate_size // 128)
        routes = active_m * Int32(self.top_k)
        unit = cta * warps_per_cta + warp
        stride = grid_x * warps_per_cta
        elem = lane * Int32(4)
        while unit < routes * iblocks:
            route = unit // iblocks
            block = unit - route * iblocks
            gid = global_ids[route].to(Int32)
            descriptor = Int32(-1)
            if gid >= Int32(0) and gid < Int32(self.map_slots):
                descriptor = tier_map[gid].to(Int32)
            if descriptor >= Int32(0):
                col = block * Int32(128) + elem
                gbase = route * Int32(2 * self.intermediate_size) + col
                ubase = gbase + Int32(self.intermediate_size)
                obase = route * Int32(self.intermediate_size) + col
                g0 = fc1_flat[gbase + Int32(0)].to(cutlass.Float32)
                g1 = fc1_flat[gbase + Int32(1)].to(cutlass.Float32)
                g2 = fc1_flat[gbase + Int32(2)].to(cutlass.Float32)
                g3 = fc1_flat[gbase + Int32(3)].to(cutlass.Float32)
                u0 = fc1_flat[ubase + Int32(0)].to(cutlass.Float32)
                u1 = fc1_flat[ubase + Int32(1)].to(cutlass.Float32)
                u2 = fc1_flat[ubase + Int32(2)].to(cutlass.Float32)
                u3 = fc1_flat[ubase + Int32(3)].to(cutlass.Float32)
                o0 = cutlass.Float32(0.0)
                o1 = cutlass.Float32(0.0)
                o2 = cutlass.Float32(0.0)
                o3 = cutlass.Float32(0.0)
                if descriptor >= Int32(256):
                    expert = descriptor & Int32(0xFF)
                    sbase = expert * Int32(3 * self.intermediate_size) + col
                    gh = self.tier1._had128_quad(g0, g1, g2, g3, lane)
                    uh = self.tier1._had128_quad(u0, u1, u2, u3, lane)
                    isz = Int32(self.intermediate_size)
                    v0 = self.tier1._silu_f32(gh[0] * rotations_flat[sbase + Int32(0)].to(cutlass.Float32)) * (uh[0] * rotations_flat[sbase + isz + Int32(0)].to(cutlass.Float32)) * rotations_flat[sbase + isz + isz + Int32(0)].to(cutlass.Float32)
                    v1 = self.tier1._silu_f32(gh[1] * rotations_flat[sbase + Int32(1)].to(cutlass.Float32)) * (uh[1] * rotations_flat[sbase + isz + Int32(1)].to(cutlass.Float32)) * rotations_flat[sbase + isz + isz + Int32(1)].to(cutlass.Float32)
                    v2 = self.tier1._silu_f32(gh[2] * rotations_flat[sbase + Int32(2)].to(cutlass.Float32)) * (uh[2] * rotations_flat[sbase + isz + Int32(2)].to(cutlass.Float32)) * rotations_flat[sbase + isz + isz + Int32(2)].to(cutlass.Float32)
                    v3 = self.tier1._silu_f32(gh[3] * rotations_flat[sbase + Int32(3)].to(cutlass.Float32)) * (uh[3] * rotations_flat[sbase + isz + Int32(3)].to(cutlass.Float32)) * rotations_flat[sbase + isz + isz + Int32(3)].to(cutlass.Float32)
                    o0, o1, o2, o3 = self.tier1._had128_quad(v0, v1, v2, v3, lane)
                else:
                    o0 = (self.tier0._cast_elem(self.tier0._silu_f32(g0)) * self.tier0._cast_elem(u0)).to(cutlass.Float32)
                    o1 = (self.tier0._cast_elem(self.tier0._silu_f32(g1)) * self.tier0._cast_elem(u1)).to(cutlass.Float32)
                    o2 = (self.tier0._cast_elem(self.tier0._silu_f32(g2)) * self.tier0._cast_elem(u2)).to(cutlass.Float32)
                    o3 = (self.tier0._cast_elem(self.tier0._silu_f32(g3)) * self.tier0._cast_elem(u3)).to(cutlass.Float32)
                activated_flat[obase + Int32(0)] = self.tier0._cast_elem(o0)
                activated_flat[obase + Int32(1)] = self.tier0._cast_elem(o1)
                activated_flat[obase + Int32(2)] = self.tier0._cast_elem(o2)
                activated_flat[obase + Int32(3)] = self.tier0._cast_elem(o3)
            unit += stride

    @cute.jit
    def _reduce_routes(
        self,
        route_output: cute.Tensor,
        output: cute.Tensor,
        down_svh: cute.Tensor,
        global_ids: cute.Tensor,
        tier_map: cute.Tensor,
        topk_weights: cute.Tensor,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_m: Int32,
    ):
        lane = tid & Int32(31)
        warp = tid >> Int32(5)
        warps_per_cta = Int32(self.cta_threads // 32)
        hblocks = Int32(self.hidden_size // 128)
        unit = cta * warps_per_cta + warp
        stride = grid_x * warps_per_cta
        elem = lane * Int32(4)
        while unit < active_m * hblocks:
            token = unit // hblocks
            block = unit - token * hblocks
            col = block * Int32(128) + elem
            acc0 = cutlass.Float32(0.0)
            acc1 = cutlass.Float32(0.0)
            acc2 = cutlass.Float32(0.0)
            acc3 = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(self.top_k):
                route = token * Int32(self.top_k) + Int32(k)
                gid = global_ids[route].to(Int32)
                descriptor = Int32(-1)
                if gid >= Int32(0) and gid < Int32(self.map_slots):
                    descriptor = tier_map[gid].to(Int32)
                if descriptor >= Int32(0):
                    rbase = route * Int32(self.hidden_size) + col
                    v0 = route_output[rbase + Int32(0)].to(cutlass.Float32)
                    v1 = route_output[rbase + Int32(1)].to(cutlass.Float32)
                    v2 = route_output[rbase + Int32(2)].to(cutlass.Float32)
                    v3 = route_output[rbase + Int32(3)].to(cutlass.Float32)
                    if descriptor >= Int32(256):
                        expert = descriptor & Int32(0xFF)
                        scale = expert * Int32(self.hidden_size) + col
                        hv = self.tier1._had128_quad(v0, v1, v2, v3, lane)
                        v0 = hv[0] * down_svh[scale + Int32(0)].to(cutlass.Float32)
                        v1 = hv[1] * down_svh[scale + Int32(1)].to(cutlass.Float32)
                        v2 = hv[2] * down_svh[scale + Int32(2)].to(cutlass.Float32)
                        v3 = hv[3] * down_svh[scale + Int32(3)].to(cutlass.Float32)
                    weight = topk_weights[route].to(cutlass.Float32)
                    acc0 += v0 * weight
                    acc1 += v1 * weight
                    acc2 += v2 * weight
                    acc3 += v3 * weight
            obase = token * Int32(self.hidden_size) + col
            output[obase + Int32(0)] = self.tier0._cast_elem(acc0)
            output[obase + Int32(1)] = self.tier0._cast_elem(acc1)
            output[obase + Int32(2)] = self.tier0._cast_elem(acc2)
            output[obase + Int32(3)] = self.tier0._cast_elem(acc3)
            unit += stride

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_s13: cute.Tensor,
        t0_s2: cute.Tensor,
        t0_g13: cute.Tensor,
        t0_g2: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_s13: cute.Tensor,
        t1_s2: cute.Tensor,
        t1_g13: cute.Tensor,
        t1_g2: cute.Tensor,
        global_ids: cute.Tensor,
        tier_map: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        route_output: cute.Tensor,
        output: cute.Tensor,
        topk_ptr: cute.Pointer,
        fc1_tmp: cute.Tensor,
        fc2_tmp: cute.Tensor,
        workspace: cute.Tensor,
        rotation_a_gate: cute.Tensor,
        rotation_a_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        down_svh: cute.Tensor,
        active_m: cutlass.Int32,
        grid_x: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        a = cute.make_tensor(a_ptr, layout=cute.make_layout((active_m * Int32(self.hidden_size),), stride=(1,)))
        topk = cute.make_tensor(topk_ptr, layout=cute.make_layout((active_m * Int32(self.top_k),), stride=(1,)))
        self.kernel(
            a, t0_w13, t0_w2, t0_s13, t0_s2, t0_g13, t0_g2,
            t1_w13, t1_w2, t1_s13, t1_s2, t1_g13, t1_g2,
            global_ids, tier_map, fc1, activated, route_output, output, topk,
            fc1_tmp, fc2_tmp, workspace, rotation_a_gate, rotation_a_up,
            gate_suh, up_suh, intermediate_rotations, down_svh, active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a: cute.Tensor,
        t0_w13: cute.Tensor,
        t0_w2: cute.Tensor,
        t0_s13: cute.Tensor,
        t0_s2: cute.Tensor,
        t0_g13: cute.Tensor,
        t0_g2: cute.Tensor,
        t1_w13: cute.Tensor,
        t1_w2: cute.Tensor,
        t1_s13: cute.Tensor,
        t1_s2: cute.Tensor,
        t1_g13: cute.Tensor,
        t1_g2: cute.Tensor,
        global_ids: cute.Tensor,
        tier_map: cute.Tensor,
        fc1: cute.Tensor,
        activated: cute.Tensor,
        route_output: cute.Tensor,
        output: cute.Tensor,
        topk: cute.Tensor,
        fc1_tmp: cute.Tensor,
        fc2_tmp: cute.Tensor,
        workspace: cute.Tensor,
        rotation_a_gate: cute.Tensor,
        rotation_a_up: cute.Tensor,
        gate_suh: cute.Tensor,
        up_suh: cute.Tensor,
        intermediate_rotations: cute.Tensor,
        down_svh: cute.Tensor,
        active_m: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        grid_raw, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        cta = Int32(bidx)
        grid_x = Int32(grid_raw)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024]

        storage = smem.allocate(Storage)
        smem_base = _base.shared_ptr_to_u32(storage.words.data_ptr())

        self._rotate_tail_inputs(
            a, rotation_a_gate, rotation_a_up, gate_suh, up_suh,
            global_ids, tier_map, tid, cta, grid_x, active_m,
        )
        self.tier0._grid_barrier(workspace, tid, grid_x)

        fc1_emit = partial(
            self._emit_tile, True, a, rotation_a_gate, rotation_a_up,
            t0_w13, t0_s13, t0_g13, t1_w13, t1_s13, t1_g13,
            fc1, global_ids, tier_map, topk, fc1_tmp, workspace,
            smem_base, tid, active_m,
        )
        self.tier0.fc1._run_persistent_gemm(
            a, a, t0_w13, fc1, t0_s13, t0_g13,
            global_ids, global_ids, global_ids, topk, fc1_tmp, workspace,
            smem_base, tid, cta, grid_x, active_m, fc1_emit,
        )
        self.tier0._grid_barrier(workspace, tid, grid_x)
        self._activate_routes(
            fc1, activated, intermediate_rotations, global_ids, tier_map,
            tid, cta, grid_x, active_m,
        )
        self.tier0._grid_barrier(workspace, tid, grid_x)

        fc2_emit = partial(
            self._emit_tile, False, activated, activated, activated,
            t0_w2, t0_s2, t0_g2, t1_w2, t1_s2, t1_g2,
            route_output, global_ids, tier_map, topk, fc2_tmp, workspace,
            smem_base, tid, active_m * Int32(self.top_k),
        )
        self.tier0.fc2._run_persistent_gemm(
            activated, activated, t0_w2, route_output, t0_s2, t0_g2,
            global_ids, global_ids, global_ids, topk, fc2_tmp, workspace,
            smem_base, tid, cta, grid_x, active_m * Int32(self.top_k), fc2_emit,
        )
        self.tier0._grid_barrier(workspace, tid, grid_x)
        self._reduce_routes(
            route_output, output, down_svh, global_ids, tier_map, topk,
            tid, cta, grid_x, active_m,
        )


_CACHE: dict[tuple[object, ...], W4A16TrellisHybridCompileResult] = {}


def _weight_elements(*, experts: int, n: int, k: int, layout: str, bits: int) -> int:
    if layout == "trellis3_t256":
        return int(experts) * (int(k) // 16) * (int(n) // 16) * (8 * int(bits))
    return _base._w4a16_weight_flat_elements(
        num_experts=experts, size_n=n, size_k=k, weight_layout=layout
    )


def compile_w4a16_trellis_hybrid(
    *,
    size_m: int,
    hidden_size: int,
    intermediate_size: int,
    tier0_num_experts: int,
    tier1_num_experts: int,
    top_k: int,
    map_slots: int,
    sms: int,
    max_shared_mem: int,
    force_tile_config: tuple[int, int, int, int],
    trellis_bits: int = 3,
) -> W4A16TrellisHybridCompileResult:
    if int(size_m) < 1 or int(size_m) > 8:
        raise ValueError("mixed Trellis v1 admits decode capacities 1..8 only")
    if int(trellis_bits) not in (3, 4, 5, 6):
        raise ValueError("trellis_bits must be one of 3, 4, 5, 6")
    fc1_k, fc1_n, fc2_k, fc2_n = (int(v) for v in force_tile_config)
    max_blocks = int(size_m) * int(top_k)

    def make_tier(experts: int, *, trellis: bool) -> _base.W4A16FusedMoeKernel:
        return _base.W4A16FusedMoeKernel(
            size_m=int(size_m),
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            num_experts=int(experts),
            top_k=int(top_k),
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            fc1_tile_n=fc1_n,
            fc1_tile_k=fc1_k,
            fc2_tile_n=fc2_n,
            fc2_tile_k=fc2_k,
            moe_block_size=8,
            max_m_blocks=max_blocks,
            element_dtype="bf16",
            fast_math=True,
            weight_layout="trellis3_t256" if trellis else "packed",
            scale_format="e4m3_k32" if trellis else "e4m3_k16",
            w13_layout="trellis3_t256_proj" if trellis else "packed",
            trellis_bits=int(trellis_bits),
            direct_topk_routes=True,
            tc_decode_fused_sum=False,
            schedule_whole_tiles=True,
            intermediate_rotation=trellis,
            route_major_a=trellis,
            defer_router_weight=True,
        )

    kernel = W4A16TrellisHybridKernel(
        tier0=make_tier(tier0_num_experts, trellis=False),
        tier1=make_tier(tier1_num_experts, trellis=True),
        map_slots=int(map_slots),
    )
    if kernel.shared_words * 4 > int(max_shared_mem) - 512:
        raise ValueError("mixed Trellis shared memory exceeds the device limit")
    device = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    key = ("w4a16_trellis_hybrid", device, kernel.__cache_key__)
    if key in _CACHE:
        return replace(_CACHE[key], size_m=int(size_m), max_m_blocks=max_blocks)

    dtype = cutlass.BFloat16
    fc1_cols = int(intermediate_size) * 2
    compile_m = _base._fake_m_for_specialization(int(size_m))
    routes = compile_m * int(top_k)

    def fake_weight(experts: int, n: int, k: int, layout: str):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (_weight_elements(experts=experts, n=n, k=k, layout=layout, bits=trellis_bits),),
            assumed_align=16,
        )

    def fake_scale(experts: int, n: int, k: int, fmt: str):
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (_base._scale_fake_int32_elements(num_experts=experts, size_k=k, size_n=n, scale_format=fmt),),
            assumed_align=16,
        )

    def fake(dtype_, elements: int, align: int = 16):
        return cute.runtime.make_fake_compact_tensor(dtype_, (int(elements),), assumed_align=align)

    t0 = kernel.tier0
    t1 = kernel.tier1
    scratch = max(fc1_cols * routes, int(hidden_size) * routes, 4 * 256 * 8 * 256)
    args = (
        make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
        fake_weight(t0.num_experts, fc1_cols, hidden_size, t0.weight_layout),
        fake_weight(t0.num_experts, hidden_size, intermediate_size, t0.weight_layout),
        fake_scale(t0.num_experts, fc1_cols, hidden_size, t0.scale_format),
        fake_scale(t0.num_experts, hidden_size, intermediate_size, t0.scale_format),
        fake(cutlass.Float32, t0.num_experts),
        fake(cutlass.Float32, t0.num_experts),
        fake_weight(t1.num_experts, fc1_cols, hidden_size, t1.weight_layout),
        fake_weight(t1.num_experts, hidden_size, intermediate_size, t1.weight_layout),
        fake_scale(t1.num_experts, fc1_cols, hidden_size, t1.scale_format),
        fake_scale(t1.num_experts, hidden_size, intermediate_size, t1.scale_format),
        fake(cutlass.Float32, t1.num_experts),
        fake(cutlass.Float32, t1.num_experts),
        fake(cutlass.Int32, routes),
        fake(cutlass.Int32, map_slots),
        fake(dtype, routes * fc1_cols),
        fake(dtype, routes * intermediate_size),
        fake(dtype, routes * hidden_size),
        fake(dtype, compile_m * hidden_size),
        make_ptr(cutlass.Float32, 4, cute.AddressSpace.gmem, assumed_align=4),
        fake(cutlass.Float32, scratch),
        fake(cutlass.Float32, scratch),
        fake(cutlass.Int32, 4 * 256 + 2),
        fake(dtype, routes * hidden_size),
        fake(dtype, routes * hidden_size),
        fake(cutlass.Float16, t1.num_experts * hidden_size),
        fake(cutlass.Float16, t1.num_experts * hidden_size),
        fake(cutlass.Float16, t1.num_experts * 3 * intermediate_size),
        fake(cutlass.Float16, t1.num_experts * hidden_size),
        1,
        1,
        current_cuda_stream(),
    )
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
    compiled = sparkinfer_compile(
        kernel,
        *args,
        compile_spec=KernelCompileSpec.from_key(
            "moe.w4a16.trellis_hybrid", W4A16TrellisHybridKernel.ABI_VERSION, key
        ),
    )
    registers = -1
    local_bytes = -1
    resources = _base._query_w4a16_kernel_resources(compiled)
    if resources is not None:
        _, registers, local_bytes = resources
        if local_bytes != 0:
            raise RuntimeError(
                f"mixed Trellis kernel spills {local_bytes} local bytes/thread"
            )
    result = W4A16TrellisHybridCompileResult(
        compiled=compiled,
        size_m=int(size_m),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        top_k=int(top_k),
        activation="silu",
        element_dtype="bf16",
        fast_math=True,
        map_slots=int(map_slots),
        tier0_num_experts=t0.num_experts,
        tier1_num_experts=t1.num_experts,
        fc1_tile_n=fc1_n,
        fc1_tile_k=fc1_k,
        fc2_tile_n=fc2_n,
        fc2_tile_k=fc2_k,
        moe_block_size=8,
        max_m_blocks=max_blocks,
        cta_threads=kernel.cta_threads,
        blocks_per_sm=kernel.blocks_per_sm,
        shared_memory_bytes=kernel.shared_words * 4,
        schedule_whole_tiles=True,
        direct_topk_routes=True,
        tc_decode_fused_sum=False,
        registers_per_thread=registers,
        local_memory_bytes=local_bytes,
        trellis_bits=int(trellis_bits),
    )
    _CACHE[key] = result
    return result


def run_w4a16_trellis_hybrid(
    a: torch.Tensor,
    prepared_tier0,
    prepared_tier1,
    topk_weights: torch.Tensor,
    global_topk_ids: torch.Tensor,
    tier_local_map: torch.Tensor,
    *,
    launch: W4A16TrellisHybridCompileResult,
    fc1: torch.Tensor,
    activated: torch.Tensor,
    route_output: torch.Tensor,
    output: torch.Tensor,
    fc1_tmp: torch.Tensor,
    fc2_tmp: torch.Tensor,
    workspace: torch.Tensor,
    rotation_a_gate: torch.Tensor,
    rotation_a_up: torch.Tensor,
    gate_suh: torch.Tensor,
    up_suh: torch.Tensor,
    intermediate_rotations: torch.Tensor,
    down_svh: torch.Tensor,
) -> torch.Tensor:
    if a.dtype != torch.bfloat16 or a.ndim != 2 or not a.is_contiguous():
        raise ValueError("mixed Trellis input must be contiguous rank-2 bf16")
    m = int(a.shape[0])
    if not 1 <= m <= launch.size_m:
        raise ValueError(f"mixed Trellis m={m} exceeds capacity {launch.size_m}")
    if int(a.shape[1]) != launch.hidden_size:
        raise ValueError("mixed Trellis hidden size mismatch")
    routes = m * launch.top_k
    if topk_weights.dtype != torch.float32 or topk_weights.numel() != routes:
        raise ValueError("mixed Trellis top-k weights must be fp32 [m, topk]")
    if global_topk_ids.dtype != torch.int32 or global_topk_ids.numel() != routes:
        raise ValueError("mixed Trellis ids must be int32 [m, topk]")
    if tier_local_map.dtype != torch.int32 or tier_local_map.numel() < launch.map_slots:
        raise ValueError("mixed Trellis descriptor map is invalid")
    if prepared_tier0.weight_layout != "packed":
        raise ValueError("mixed Trellis tier0 weights are not packed NVFP4")
    if prepared_tier1.weight_layout != "trellis3_t256":
        raise ValueError("mixed Trellis tier1 weights are not Trellis256")
    if int(prepared_tier1.trellis_bits) != launch.trellis_bits:
        raise ValueError("mixed Trellis bitrate does not match launch")

    device = a.device
    tensors = (
        topk_weights, global_topk_ids, tier_local_map, fc1, activated,
        route_output, output, fc1_tmp, fc2_tmp, workspace, rotation_a_gate,
        rotation_a_up, gate_suh, up_suh, intermediate_rotations, down_svh,
    )
    if any(t.device != device or not t.is_contiguous() for t in tensors):
        raise ValueError("mixed Trellis runtime tensors must be contiguous on input device")
    capacity_routes = launch.size_m * launch.top_k
    if fc1.numel() < capacity_routes * 2 * launch.intermediate_size:
        raise ValueError("mixed Trellis FC1 arena is undersized")
    if activated.numel() < capacity_routes * launch.intermediate_size:
        raise ValueError("mixed Trellis activation arena is undersized")
    if route_output.numel() < capacity_routes * launch.hidden_size:
        raise ValueError("mixed Trellis route output arena is undersized")
    if output.dtype != torch.bfloat16 or output.numel() < m * launch.hidden_size:
        raise ValueError("mixed Trellis output arena is invalid")
    if rotation_a_gate.dtype != torch.bfloat16 or rotation_a_up.dtype != torch.bfloat16:
        raise TypeError("mixed Trellis rotated-A arenas must be bf16")
    for name, value, expected in (
        ("gate_suh", gate_suh, launch.tier1_num_experts * launch.hidden_size),
        ("up_suh", up_suh, launch.tier1_num_experts * launch.hidden_size),
        ("intermediate_rotations", intermediate_rotations, launch.tier1_num_experts * 3 * launch.intermediate_size),
        ("down_svh", down_svh, launch.tier1_num_experts * launch.hidden_size),
    ):
        if value.dtype != torch.float16 or value.numel() < expected:
            raise ValueError(f"mixed Trellis {name} is invalid")

    def weights(prepared):
        return prepared.w13.view(torch.int32).view(-1), prepared.w2.view(torch.int32).view(-1)

    t0_w13, t0_w2 = weights(prepared_tier0)
    t1_w13, t1_w2 = weights(prepared_tier1)
    props = torch.cuda.get_device_properties(device)
    grid_x = _base._w4a16_fused_persistent_grid_x(
        fused=launch,
        m=m,
        topk=launch.top_k,
        intermediate_size=launch.intermediate_size,
        activation="silu",
        direct_topk_routes=True,
        sms=int(props.multi_processor_count),
    )
    stream = current_cuda_stream()
    launch.compiled(
        make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
        t0_w13,
        t0_w2,
        prepared_tier0.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier0.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier0.w13_global_scale,
        prepared_tier0.w2_global_scale,
        t1_w13,
        t1_w2,
        prepared_tier1.w13_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier1.w2_scale.view(torch.uint8).view(torch.int32).view(-1),
        prepared_tier1.w13_global_scale,
        prepared_tier1.w2_global_scale,
        global_topk_ids.view(-1),
        tier_local_map,
        fc1.view(-1),
        activated.view(-1),
        route_output.view(-1),
        output.view(-1),
        make_ptr(cutlass.Float32, topk_weights.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
        fc1_tmp,
        fc2_tmp,
        workspace,
        rotation_a_gate.view(-1),
        rotation_a_up.view(-1),
        gate_suh.view(-1),
        up_suh.view(-1),
        intermediate_rotations.view(-1),
        down_svh.view(-1),
        m,
        grid_x,
        stream,
    )
    return output[:m]


__all__ = [
    "W4A16TrellisHybridCompileResult",
    "compile_w4a16_trellis_hybrid",
    "run_w4a16_trellis_hybrid",
]
