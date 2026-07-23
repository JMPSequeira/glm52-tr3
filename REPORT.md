# Optimization Report

## Scope

Optimize prefill and decode for `brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid` on four RTX PRO 6000 Blackwell GPUs while preserving:

- the full 1,048,576-token model context;
- an exact 1,048,576-token global KV budget;
- TP4/DCP4 execution;
- no MTP in the primary recipe;
- admission for eight sequences;
- C1 performance at 0, 16k, 32k, and 64k context as the comparison metric.

More than 140 controlled runs were used to isolate scheduler, KV transport, source, and kernel effects. Every retained behavioral change was checked against exact capacity and startup invariants; new mixed-kernel paths received layer/rank numerical parity checks before throughput comparison.

## Final retained result

| Context | Stock prefill | Final prefill | Change | Stock decode | Final decode | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 0 / 8k | 2,038 | **2,523** | +23.8% | 22.384 | **46.240** | +106.6% |
| 16k | 2,001 | **2,457** | +22.8% | 18.702 | **46.228** | +147.2% |
| 32k | 1,638 | **2,447** | +49.4% | 19.163 | **45.738** | +138.7% |
| 64k | 1,496 | **2,397** | +60.2% | 18.765 | **45.021** | +139.9% |
| Geometric mean | 1,777.97 | **2,455.59** | **+38.1%** | 19.697 | **45.804** | **+132.5%** |

All throughput values are tokens/second. Prefill uses 8k for the first row; decode uses zero prior context.

The clean source-built publication image reproduced the C1 result at 2,466.84 prefill and 45.840 decode tok/s geometric mean (+0.46% / +0.08% versus the retained image). It also admitted eight simultaneous zero-context streams and sustained **206.006 aggregate tok/s** over a 30-second C8 validation. C8 was an admission constraint, not the primary optimization metric.

## BF16-reference quality evidence

Checkpoint author **Brandon M. Music** published an end-to-end teacher-forced comparison against stored base-model BF16 full-vocabulary logits. The production `nvfp4_ds_mla` candidate produced mean KLD runs **0.146681 / 0.145467 / 0.146602 / 0.151969 / 0.154528**, for a mean of **0.149049**, sample SD **0.003968**, and range **0.145467–0.154528**.

The protocol used one fixed 2,048-token `Salesforce/wikitext` window per fresh model boot, 2,047 scored positions, vocabulary size 154,880, stride 512, TP4/DCP4, and speculative decoding off. The [immutable source artifact](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid/blob/1d10e2114aa8a3f0bde44809808bbddee168c93a/benchmarks/2026-07-18/kld-bf16-reference.json) calls this teacher-forced logit divergence and cautions that it is one fixed window.

Interpretation boundary: this measures the assembled hybrid checkpoint and serving path, including compressed KV. It does **not** isolate weight-only NVFP4/TR3 error. A second FP8-KV series measured mean KLD 0.137302, confirming that KV format changes the result. The evaluator, stored teacher logits, exact window, and KL direction were not published, so this repository does not claim independent reproduction of the metric.

## What worked

### 1. Restore the intended dispatch and memory envelope

The first large gains came from configuration rather than a new kernel:

- enable Trellis256 beginning at `M=1` instead of waiting for larger batches;
- raise the compressed-KV gather ceiling from 16k to 64k;
- use 64-token KV blocks and exactly 4,096 global blocks;
- increase no-MTP scheduled tokens while staying below the first-request conversion OOM boundary;
- retain graph ceiling 16 and C8 admission.

A 6,144-token scheduler was fast on the older stack, but the final mixed stack needed a 5,120-token ceiling to leave enough first-request workspace. This is the memory-safe retained value.

### 2. Move to the pinned Gilded Gnosis v19 stack

Once its launch settings matched the workload, the v19 CUDA 13.2/B12X stack preserved stock-class prefill and raised decode to about 31.8 tok/s geometric mean. The important pieces were the SM120 W4A16 implementation, calibrated NVFP4 MLA scales, Trellis256 at small `M`, and 64k compressed-KV gather.

### 3. Planned Trellis execution and stream overlap

The checkpoint's kept NVFP4 experts and Trellis tail had independent work. A planned Trellis API made scratch sizing explicit and rejected unsupported layouts before compute. Running the two tiers on separate CUDA streams with preallocated per-layer events raised no-MTP decode from about 31.8 to 38.0 tok/s while preserving capacity.

Production parity across 75 routed layers and four ranks reached maximum relative L2 error 0.00254 and minimum cosine 0.9999969 for the planned path.

### 4. Persistent mixed NVFP4/Trellis MoE

Profiling showed routed W4A16 MoE as the largest target-model decode cost. The retained kernel routes 64 NVFP4 and 192 Trellis experts inside one persistent grid, keeps route-major FC1/FC2 workspaces, applies rotations only to the Trellis tier, and performs one final mixed top-k reduction.

The path reuses the existing planned-decode arena rather than allocating another full workspace. M=1 and M=4 production parity covered 600 layer/rank comparisons; maximum relative error stayed below 0.00573 and minimum cosine above 0.999983.

### 5. Shape-specific tile selection

The final no-MTP target cycle favored 128×128 tiles. This raised decode from the first mixed-kernel result's 41.0 tok/s geometric mean to 45.8 tok/s without materially changing prefill. MTP verifier shapes instead favor 64×256 tiles; the launch recipes set these independently through an environment-controlled tile tuple.

### 6. Causal MTP verifier routing

The draft model was not the dominant MTP cost. Profiling measured roughly 2.5 ms for all proposals but about 38–40 ms for target verification. Routing causal multi-row verification through B12X sparse decode and absorbing the MXFP8 MLA BMM materially improved C1:

- retained MTP3: 88.40 tok/s decode geometric mean;
- retained C1-specialized MTP4: **90.29 tok/s** decode geometric mean;
- MTP4 matrix: 90.507 / 90.995 / 90.836 / 88.849 tok/s at 0 / 16k / 32k / 64k.

MTP4 is optional because it did not retain concurrent-service throughput.

### 7. Long-context spot qualification

A deterministic prompt was calibrated through `/tokenize` to exactly 300,000 tokens with a needle at 50%. Both forced verifier-to-B12X decode and the safe extend control returned exactly `KITE-7391-ONYX` from the same prompt SHA-256. This did not reproduce the reported 300k retrieval failure, but it remains one prompt and one position rather than exhaustive 1M-context qualification.

### 8. Padded MTP prefill workspace correctness

The first exact 128k MTP4 request exposed a B12X DCP projection contract bug, not a KV-capacity failure. Speculative scheduling produced a visible 2,047-token prefill chunk while the B12X head-major output retained its safe 2,048-row aligned pitch. The old validator required a compact stride and killed the engine before projection, even though projection immediately compacts the view before cuBLAS.

The publication patch now validates the actual invariant: a unit inner stride, a token stride equal to the latent rank, and an integral head pitch large enough to cover every visible token row. Workspace provenance is still checked separately. An exact 131,072-token C1 retrieval request completed successfully. A 128k C4 sustained check then reported average/max running requests of 4/4, zero queue, zero request errors, and 39.647 aggregate tok/s. This qualifies correctness and admission only; it does not change the MTP3 concurrent-performance decision.

## What did not work

### Oversized scheduler batches

An 8,192-token scheduler exhausted VRAM on the first 8k request. On the mixed stack, 6,144 also left insufficient conversion/indexer workspace. These were real memory failures, not throughput regressions; they were rejected.

### Blind source upgrades

Moving to newer vLLM/SparkInfer source without preserving workload-specific dispatch improved some prefill shapes but regressed decode by about 14%. A later exact v20 source transfer was neutral: +0.24% prefill and -0.05% decode geometric mean. Source age was not the bottleneck.

### Pure EXL3 backend integration

vLLM PR #139's pure EXL3 backend was valuable upstream work but did not replace this checkpoint's mixed NVFP4/TR3 backend. The target quantization metadata deliberately selects the hybrid loader first. A full-EXL3 comparison image demonstrated an interesting 8k prefill contrast, but its quantization, MTP depth, hardware, KV capacity, and 64k behavior were not apples-to-apples. No result was claimed from that transfer.

### DCP query splitting and speculative KV transport

Enabling DCP query split changed prefill by only +0.52% geometric mean because the sparse-indexer query slice was already a small fraction of the cycle. Deeper compressed-KV prefetch, sparse-decode transport experiments, and an extra lookahead workspace either measured neutral or reduced memory headroom. None were retained.

### Tile sweeps without a new schedule

Multiple 64×128, 64×256, 128×64, FC1-only, FC2-only, blocks-per-SM, local-argmax, and alternate reduction variants failed to beat the shape-matched retained tuples. Recompiling the same schedule for exact M=1 was neutral. The remaining MoE opportunity likely needs a different route-row decomposition or serialized weight order, not another tuple sweep.

### Deep speculative decoding

MTP5 was close but not better than MTP4. MTP6 and MTP7 fell below the break-even point as acceptance stopped amortizing extra proposals; their decode geometric means dropped to about 71.4 and 68.5 tok/s. More draft depth was not free throughput.

### MTP4 under C4 concurrency

MTP4 was the best C1 depth but collapsed to roughly 36 aggregate tok/s at C4, versus 204.5 tok/s for the retained MTP3 C4 recipe. The optional MTP4 launcher is therefore labeled C1-specific rather than a universal production default.

### Cooperative whole-grid admission as a speed change

SparkInfer's cooperative-launch safeguard is useful for barrier-bearing kernel correctness. Real-weight parity passed, but the matched MTP4 cycle changed from 34.568999 to 34.567257 ms (-0.005%), and the no-MTP A/B coherently regressed decode by 0.68%. It was rejected as a performance optimization.

### P2P module override

The workstation did not have NVIDIA's optional P2P module override enabled. Applying it requires root, stopping GPU workloads, and reloading the driver or rebooting. It remains a separate host-level experiment and is not presented as a code result.

## Profiling conclusions

The optimization sequence followed measured costs:

1. Long prefill initially spent substantial time in PCIe collectives and TR3 tail dispatch.
2. The retained v19 no-MTP cycle spent about 30% in its two routed W4A16 launches.
3. Planned execution removed avoidable serialization between the NVFP4 and Trellis tiers.
4. The persistent mixed kernel removed the remaining split-tier launch/reduction overhead.
5. Under MTP, target verification—not proposal generation—set the cycle time.

This is why scheduler tuning, route selection, and the mixed routed-MoE kernel transferred, while unrelated source upgrades and small indexer changes did not.

## Final decision

- **Default:** no MTP, C8 admission, 5,120 scheduled tokens, graph16, exact 1M model/KV capacity, planned prefill/tail, persistent mixed MoE with 128×128 tiles.
- **Optional C1 mode:** greedy MTP4, C4 admission, 2,048 scheduled tokens, graph16, verifier decode through Q=5, absorbed MXFP8 MLA BMM, mixed MoE with 64×256 tiles.
- **Not claimed:** exhaustive 1M retrieval correctness, performance on other GPUs, MTP4 concurrent-service superiority, or gains from rejected upstream/source experiments.
