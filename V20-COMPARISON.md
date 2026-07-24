# Runtime divergence from v20

## Scope

This document compares `glm52-tr3:runtime-v3` with both relevant meanings of v20:

1. Canonical Gilded Gnosis v20:
   `voipmonitor/vllm:gilded-gnosis-v20-vllm3e731bc-si1a88b38-fi801d57a-cu132-20260722`
2. VerdictAI's full-EXL3 v20 overlay:
   `verdictai/glm52-exl3-sparkinfer:v20-gg6722c1d-si1a88b38-cu132-sm120a`

The local comparisons used the published image digests and the installed source trees. Performance comparisons refer to the controlled runs linked below.

## Executive summary

Our runtime is not an older alternative to v20. It uses the same v20 core source revisions:

| Component | Our runtime | v20 |
|---|---|---|
| vLLM source | `3e731bc043d2` | `3e731bc043d2` |
| SparkInfer base | `1a88b389a8d1` | `1a88b389a8d1` |
| Torch | `2.12.0+cu132` | `2.12.0+cu132` |
| CUTLASS DSL | `4.6.0` | `4.6.0` |
| Quack kernels | `0.6.1` | `0.6.1` |
| CUDA | `13.2.1` | `13.2.1` |

The material divergences are:

1. mixed NVFP4/TR3 checkpoint support;
2. persistent mixed-MoE kernels;
3. planned Trellis execution and tier overlap;
4. fused BF16 MLA query projection and assembly;
5. compressed-KV and exact-1M launch policy;
6. dynamic concurrency-aware MTP serving;
7. build and provenance differences inherited from the v19 base image.

## Source-level divergence

### vLLM versus canonical v20

Comparison of the installed source trees found:

- 3,935 source files identical;
- one added file;
- ten modified files;
- approximately 2,145 added and 18 removed lines.

The added file is the 1,639-line hybrid backend:

```text
vllm/model_executor/layers/quantization/nvfp4_tr3_hybrid.py
```

The integration changes are concentrated in:

| File | Divergence |
|---|---|
| `config/model.py` | Selects mixed `nvfp4_tr3_hybrid` before pure EXL3 |
| `config/quantization.py` | Registers the hybrid quantization method |
| `quantization/__init__.py` | Imports and exposes the hybrid backend |
| `fused_moe/routed_experts.py` | Adds checkpoint-native serialized expert loading |
| `models/deepseek_v2.py` | Routes rank-sliced TR3 tensors to the hybrid expert consumer |
| `b12x_mla_sparse.py` | Accepts padded DCP output pitches, including 2,048 rows for a 2,047-row speculative prefill |
| `kernels/attention/b12x_mxfp8_bmm.py` | Adds SparkInfer capability checks, custom ops, and fused-query execution |
| `layers/attention/mla_attention.py` | Dispatches qualified BF16/MXFP8 query projection through the fused path |
| `warmup/kernel_warmup.py` | Prewarms graph-visible fused-query variants |
| `_version.py` | Differs because of build/version provenance |

The repository applies these changes through [`patches/vllm-fused-query.patch`](patches/vllm-fused-query.patch) and [`runtime/patch-vllm-hybrid.py`](runtime/patch-vllm-hybrid.py), then installs the hybrid backend from [`runtime/nvfp4_tr3_hybrid.py`](runtime/nvfp4_tr3_hybrid.py).

### SparkInfer versus canonical v20

The retained runtime diverges across 19 SparkInfer source paths:

- eight paths exist only in this runtime;
- one canonical path is relocated;
- ten paths are modified;
- the raw tree diff is 7,769 additions and 1,597 deletions, inflated by the 1,475-line MXFP8 module relocation.

The significant additions are:

- planned full-rotation Trellis APIs;
- prepared weight and scratch-buffer interfaces;
- expanded W4A16 support;
- a persistent mixed NVFP4/Trellis kernel;
- route-packing changes;
- fused BF16 MLA query projection and direct query assembly;
- separate Trellis decode and prefill plans.

The final runtime adds these modules beyond canonical v20:

```text
sparkinfer/gemm/_shared/mxfp8_bmm.py
sparkinfer/gemm/mla_query_projection/__init__.py
sparkinfer/gemm/mla_query_projection/_bf16.py
sparkinfer/gemm/mla_query_projection/api.py
sparkinfer/moe/_shared/kernels/w4a16/trellis_hybrid.py
sparkinfer/moe/trellis_moe/__init__.py
sparkinfer/moe/trellis_moe/api.py
sparkinfer/moe/trellis_moe/_impl.py
```

The Trellis and mixed-MoE changes are in [`patches/sparkinfer-final.patch`](patches/sparkinfer-final.patch); the fused-query series is in [`patches/sparkinfer-fused-query.patch`](patches/sparkinfer-fused-query.patch). The final mixed kernel is maintained separately in [`runtime/trellis_hybrid.py`](runtime/trellis_hybrid.py).

### Versus VerdictAI full-EXL3 v20

Verdict v20 already includes the planned Trellis API. The installed SparkInfer comparison now finds:

- 159 source files identical;
- our 882-line `trellis_hybrid.py` is additional;
- four fused-query modules are additional and the canonical MXFP8 BMM module is relocated under `gemm/_shared`;
- `w4a16/kernel.py` differs by 19 additions and three deletions;
- our compiler-cache implementation is 28 additions and 64 deletions behind Verdict's device-architecture-aware cache format.

The W4A16 additions make route-major layout configurable and permit deferred router weighting.

The principal vLLM backend difference is an alternative quantization implementation:

```text
Verdict v20: exl3.py                  1,804 lines
Our runtime: nvfp4_tr3_hybrid.py      1,639 lines
```

These are alternatives rather than successive versions of one backend.

## Checkpoint representation

### Verdict full-EXL3 v20

The pure EXL3 backend expects Trellis/EXL3 data for every routed expert. Its rank-sliced path uses:

- planned SparkInfer Trellis for decode-sized batches;
- the ExLlamaV3 extension for larger batches;
- one homogeneous EXL3 expert representation.

### Our runtime

The target checkpoint deliberately contains, per routed layer:

- 64 hot NVFP4 experts;
- 192 TR3/Trellis experts.

Our backend reads the tier bitmap and preserves both representations. The pure EXL3 backend cannot directly replace it because the 64 NVFP4 experts intentionally have no Trellis payload.

The controlled v20 transfer in run 138 therefore used the Verdict v20 base only after restoring the hybrid loader and kernels.

## Routed-MoE execution

| Shape | Our runtime | Pure EXL3 v20 |
|---|---|---|
| C1 target verification | One persistent mixed NVFP4/Trellis grid through M=5 | Homogeneous planned/fused EXL3 |
| C2-C4 verification | One persistent mixed grid through M=4 | Homogeneous planned/fused EXL3 |
| Above the mixed ceiling | NVFP4 kept tier plus planned Trellis tail | Planned Trellis for all experts |
| Prefill | Planned Trellis tail plus NVFP4 kept experts | Planned/fused EXL3 |
| Tier execution | Concurrent streams where applicable | No separate NVFP4/Trellis tiers |
| Reduction | One mixed FP32 top-k reduction | EXL3-only reduction |
| Unmapped routes | `-1`, skipped by the Trellis packer | No mixed-tier unmapped routes |

The route-map correction is specific to the hybrid representation: NVFP4 routes are omitted from the Trellis tail rather than mapped to fake Trellis expert zero and discarded afterward.

## Serving configuration

### Adaptive default

```text
C1: probabilistic MTP4
C2-C4: probabilistic MTP3
batch schedule: [[1,1,4],[2,4,3]]
CUDA graphs: 1,2,4,8,16,20
maximum scheduled tokens: 2,048
model/KV capacity: 1,048,576 tokens
parallelism: TP4/DCP4 A2A
```

The recipe is implemented by [`scripts/launch-mtp-dynamic.sh`](scripts/launch-mtp-dynamic.sh).

### No-MTP recipe

```text
C8 admission
5,120 scheduled tokens
graph ceiling 16
persistent mixed MoE through M=4
planned Trellis prefill and tail
exact 1,048,576 model/KV tokens
```

The recipe is implemented by [`scripts/launch-no-mtp.sh`](scripts/launch-no-mtp.sh).

### Supplied full-EXL3 v20 recipe

The supplied full-EXL3 configuration used:

- MTP3;
- 3,072 KV blocks;
- a 786,432-token capacity ceiling;
- pure EXL3 expert routing;
- query splitting and MTP-specific speculative options.

The reduced capacity is a recipe difference, not a v20 source limitation. Run 138 demonstrated that the v20 source can retain exactly 1,048,576 tokens when launched with the hybrid capacity configuration.

## Performance evidence

### Source age alone was neutral

Run 099 changed the core stack to v20 while preserving the hybrid backend:

| Metric | Before v20 source | v20 source | Change |
|---|---:|---:|---:|
| Prefill geometric mean | 2,473.13 tok/s | 2,475.09 tok/s | +0.08% |
| Decode geometric mean | 41.0352 tok/s | 41.0866 tok/s | +0.13% |

Run 138 transferred the hybrid checkpoint onto the supplied Verdict v20 base:

| Metric | Retained hybrid | Verdict-v20 base plus hybrid | Change |
|---|---:|---:|---:|
| Prefill geometric mean | 2,455.59 tok/s | 2,461.45 tok/s | +0.238% |
| Decode geometric mean | 45.804 tok/s | 45.781 tok/s | -0.051% |

Both comparisons are within run variance. The v20 source cutover was compatible but was not itself a material optimization.

### The custom kernel divergence was material

The retained 128x128 persistent mixed kernel in run 108 improved no-MTP decode by 11.482% over matched v20-source run 099:

```text
41.0866 -> 45.8039 tok/s geometric mean
```

The gain came from:

- one persistent grid for both quantization tiers;
- reduced K accumulation depth;
- shared FC1, activation, and FC2 workspaces;
- one top-k reduction;
- avoiding separate rounded tier outputs.

### Fused MLA query result

Runs 175 and 177 repeated the fused BF16 query path against matched control run 176. The average candidate decode geometric mean improved by **1.200%**:

```text
45.7798 -> 46.3293 tok/s
```

Per-context gains were **1.190% / 1.213% / 1.248% / 1.151%** at 0/16k/32k/64k. The two candidate runs differed by at most 0.193%. Averaged prefill geometric mean moved by 0.072%, so the change is decode-only.

### Current adaptive result

The final dynamic recipe measured:

| Concurrency | Aggregate throughput | Speculative cycle |
|---:|---:|---:|
| C2 | 139.993 tok/s | 21.4325 ms |
| C4 | 206.819 tok/s | 14.5939 ms |

Run 173 and its run 174 control isolate the route-map change. The conservative result is a 1.585% C4 cycles-per-second gain. The larger raw-throughput movement is acceptance-sensitive. Retained metrics are machine-readable in [`results/summary.json`](results/summary.json).

These dynamic-MTP values are not a direct comparison against pure full-EXL3 v20 because the quantized checkpoint, capacity, and launch recipe differ.

## Build and provenance differences

The public build starts from the v19 base image and overlays v20 source. Functionally it runs the v20 core source, but some inherited metadata remains stale:

- `importlib.metadata.version("vllm")` reports the v19 build string;
- the API system fingerprint consequently reports v19;
- inherited labels claim B12X `4cfa530` and CUTLASS DSL `4.5.3`;
- the actual installed source/packages are SparkInfer `1a88b389` and CUTLASS DSL `4.6.0`.

Canonical v20 carries consistent v20 package metadata and provenance labels.

Our image also defaults to generic cache paths such as:

```text
/cache/jit/vllm
/cache/jit/triton
```

Canonical v20 namespaces caches by a source fingerprint:

```text
/cache/jit/vllm3e731bc043-b12x1a88b389a8-.../
```

The generic paths can permit stale compiled artifacts to survive a source change when the same cache volume is reused. Canonical v20's fingerprinted layout avoids that ambiguity.

### Image footprint

| Image | Uncompressed size |
|---|---:|
| Canonical GG v20 | 24.912 GB |
| Verdict full-EXL3 v20 | 25.053 GB |
| Our runtime | 25.240 GB |

Our image is approximately 328 MB larger than canonical v20, primarily because of the hybrid sources, rebuilt EXL3 extension, compatibility shim, and retained overlay layers.

## Conclusion

The runtime is best described as:

> v20 core source plus a workload-specific mixed NVFP4/TR3 backend, persistent mixed kernels, fused MLA query assembly, exact-1M capacity configuration, and dynamic concurrency-aware MTP serving.

It does not significantly diverge from v20 in general vLLM or CUDA infrastructure. It diverges at the checkpoint loader, routed-MoE execution, KV policy, and serving recipe: the areas responsible for compatibility and performance on this hybrid checkpoint.

The largest remaining maintainability gap is replacing the v19 base/overlay construction with a clean native-v20 base so that package metadata, labels, cache fingerprints, and runtime source all describe the same stack.
