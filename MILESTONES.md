# Milestone Timeline

The optimization campaign ran from 20–23 July 2026. Run numbers are preserved to make decisions traceable even though the public repository contains only compact retained metrics.

## 20 July 2026 — Baseline and constraints

| Milestone | Evidence | Decision |
|---|---|---|
| Stock baseline captured | Run 001: 1,777.97 prefill and 19.697 decode tok/s geometric means | Established the comparison point |
| Capacity invariants recorded | TP4/DCP4, 1,048,576 model/KV tokens, no MTP, C8 required | Every candidate must preserve them |
| First scheduler/dispatch sweep | Runs 007–014 | Trellis256 from M=1, 64k CKV gather, and larger scheduler chunks were useful |
| Previous stock-image optimum | Run 014: 2,045.43 prefill and 23.298 decode tok/s geometric means | +15.0% / +18.3% over stock |

## 21 July 2026 — Gilded Gnosis v19 transfer

| Milestone | Evidence | Decision |
|---|---|---|
| Initial v19 attempt exposed a prefill regression | Runs 002–006 | Regression was dispatch/configuration-induced, not intrinsic |
| Matched v19 configuration recovered prefill | Runs 018 and 020 | Retain v19 source/runtime stack |
| Decode reached 31.796 tok/s geometric mean | Run 018 | +61.4% over stock decode while maintaining stock-class prefill |
| C8 and exact 1M capacity passed | Run 022 | v19 became the new base |
| Profiling identified routed MoE as the main decode cost | Run 048 | Target W4A16 expert execution next |

## 22 July 2026 — Planned execution and persistent mixed MoE

| Milestone | Evidence | Decision |
|---|---|---|
| Planned Trellis API passed production parity | Runs 049–052 | Safe basis for fixed scratch and launch planning |
| NVFP4/Trellis stream overlap reached about 38.0 tok/s | Runs 055–056 | Retain concurrent tier execution |
| CKV lookahead and several source-only changes were neutral | Runs 057 onward | Preserve memory headroom |
| Persistent mixed kernel passed M=1/M=4 parity | Runs 086–089 | Fuse 64 NVFP4 and 192 Trellis experts into one grid |
| Mixed kernel reached 41.035 tok/s decode geometric mean | Run 089 | +7.68% over the split planned kernel |

## 23 July 2026 — Final tiles, MTP, transfer tests, qualification

| Milestone | Evidence | Decision |
|---|---|---|
| Causal MTP verifier route + mixed MoE | Runs 097–098 | Retain MTP3 for concurrent service; 88.398 C1 geometric mean and 204.5 aggregate tok/s at C4 |
| 128×128 no-MTP tile specialization | Run 108 | Final no-MTP decode: 45.804 tok/s geometric mean |
| Final no-MTP C8 smoke | Run 112 | Eight requests admitted; exact 1M capacity retained |
| MTP depth sweep | Runs 113–118 | MTP4 wins C1 at 90.293 tok/s; MTP6/7 rejected; MTP4 C4 rejected |
| Exact-M1, local reduction, and tile micro-sweeps | Runs 127–135 | No further material kernel gain |
| Current-v20 and external full-EXL3 transfer | Runs 136–138 | Neutral for the mixed checkpoint; no source cutover |
| Cooperative whole-grid admission | Runs 140–141 | Numerically valid, performance-neutral; not retained |
| 300k forced-route retrieval and safe control | Runs 142–143 | Both returned the exact needle; one-position spot check passed |
| Publication package | Source build, retained launch recipes, report, compact results | Final recipe frozen and published |

## Final state

```text
Stock
  └─ scheduler + CKV dispatch
      └─ Gilded Gnosis v19
          └─ planned Trellis execution
              └─ concurrent NVFP4/Trellis tiers
                  └─ persistent mixed MoE
                      ├─ 128×128 no-MTP / C8 retained default
                      └─ 64×256 MTP4 / C1 optional specialization
```

The campaign stopped after the remaining tested variations measured neutral, regressed throughput, reduced memory headroom, or changed the workload enough to make the comparison invalid.
