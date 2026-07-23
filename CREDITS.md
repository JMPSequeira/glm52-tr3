# Credits

This repository is an integration and optimization layer. It would not exist without the model, serving engines, kernels, benchmark tooling, and earlier GLM-5.2 enablement produced by the projects and people below.

## Optimization campaign

- **[JMPSequeira](https://github.com/JMPSequeira)** — optimization design, implementation, four-GPU benchmarking and profiling, experiment analysis, retained launch recipes, reproducible image packaging, and this publication.

## Model and architecture

- **Z.ai / GLM team** — [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), including the model architecture, long-context design, sparse attention, and MTP work.
- **Brandon / BrandonMusic** — the [`GLM-5.2-NVFP4-TR3-Hybrid`](https://huggingface.co/brandonmusic/GLM-5.2-NVFP4-TR3-Hybrid) checkpoint, its mixed NVFP4/Trellis layout and metadata, calibration assets, extensive GLM-5.2 integration work, the relevant vLLM/B12X branches, and the [SM120 ExLlamaV3 retile work](https://github.com/brandonmmusic-max/exllamav3/tree/a1-retile-sm120).
- **ExLlamaV3 contributors, led by turboderp** — the [EXL3 quantization format and CUDA implementation](https://github.com/turboderp-org/exllamav3) used by the Trellis expert tier.

## Serving and kernels

- **local-inference-lab contributors** — the [vLLM GLM-5.2 branches](https://github.com/local-inference-lab/vllm), [B12X/SparkInfer kernels](https://github.com/local-inference-lab/b12x), CUDA 13.2 Blackwell images, PCIe/NCCL work, and [`llm-inference-bench`](https://github.com/local-inference-lab/llm-inference-bench). These projects provide most of the platform on which this optimization was built and measured.
- **David Young** — [`davidsyoung/vllm-glm52`](https://github.com/davidsyoung/vllm-glm52), an important earlier public GLM-5.2 vLLM implementation and reference during bring-up.
- **vLLM contributors** — the [vLLM serving engine](https://github.com/vllm-project/vllm), scheduler, distributed execution, attention interfaces, speculative decoding, and model integration framework.
- **SparkInfer/B12X contributors** — the W4A16, sparse MLA, planned EXL3/Trellis, routing, and Blackwell kernel infrastructure modified by this repository's patch series.
- **FlashInfer contributors** — [FlashInfer](https://github.com/flashinfer-ai/flashinfer) primitives and Blackwell inference infrastructure included in the runtime stack.
- **PyTorch contributors** — [PyTorch](https://github.com/pytorch/pytorch), CUDA extension, tensor, distributed, and compiler foundations.

## NVIDIA stack

- **NVIDIA** — the RTX PRO 6000 Blackwell hardware, CUDA, NVFP4, cuBLAS, cuDNN, NCCL, CUTLASS and CUTLASS DSL, ModelOpt formats, and profiling tooling used throughout this work.
- **QuACK contributors** — [QuACK kernels](https://github.com/Dao-AILab/quack) and compilation support used by the pinned SparkInfer stack.

## Comparative and methodological references

- **VerdictAI** — the published full-EXL3 v20 runtime used for the controlled source/configuration transfer comparison.
- **NInfer contributors** — exact-shape replay and serialized-operator methodology that informed the M=1 specialization experiment.
- **Kog authors** — persistent-grid, route-ordering, prefetch, and local-handoff ideas used as a systems-design reference. No Kog performance number is presented as comparable to this model or rig.

## Attribution boundaries

The benchmark results and integration decisions in this repository apply to the stated hybrid checkpoint and four-GPU test rig. Upstream projects do not endorse these results and are not responsible for local modifications. Their names and trademarks remain their own.

If an upstream contribution is missing or described imprecisely, please open an issue so the attribution can be corrected.
