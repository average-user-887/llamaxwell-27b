# Dense v14 N3 gate, up and SwiGLU fusion

Dense v14 fuses the gate and up projections with SwiGLU for three-column dense evaluation on SM52. At a 4,096-token cached prompt, client generation improved **1.02% in ABBA and 1.31% in BAAB**. Request wall time fell **0.98% and 1.21%**, respectively. At a 77,920-token saved prefix, client generation improved **0.49%** and wall time fell **0.47%**. These are medians of paired changes under the active GPU governor on a Quadro M6000 24 GiB.

## Serving replay

| Cached prompt | Run order | Timed pairs | Client generation | Server generation | Request wall |
| --- | --- | ---: | ---: | ---: | ---: |
| 4,096 tokens | OFF, ON, ON, OFF | 8 | +1.02% | +1.03% | −0.98% |
| 4,096 tokens | ON, OFF, OFF, ON | 6 | +1.31% | +1.28% | −1.21% |
| 77,920 tokens | OFF, ON, ON, OFF | 4 | +0.49% | +0.48% | −0.47% |

All arms used the same v14 CUDA library, Qwen3.8-27B-Q4_0-head4, two 131,072-token slots, the v13 serving profile and MTP draft depth 2. Each replay restored the same saved state before a sentinel and every timed request. The 4K trials emitted 256 tokens per timed request and measured client generation over tokens 16–240. The long-prefix trials emitted 128 and measured over tokens 8–120.

All **48 request trajectories and 9,216 generated token positions** matched across corresponding arms. Checks covered generated token IDs, streamed text, prompt and cache counters, MTP proposals and acceptances, finish reason, stop word and truncation. The OFF arms witnessed the stock gate and up kernels with Q8 reuse. The ON arms witnessed the fused route with a Q8 cache fill or hit.

None of the prespecified pairs shared a stable sampled P0 clock signature. In the 4K pairs with stable samples from both arms, ON graphics clocks were higher. The observed serving gains repeated with reversed startup order under the active governor. The measurements leave fixed-clock serving attribution unresolved. The [JSON receipt](receipts.json) records aggregate clock counts, directional timing medians and build pins.

## Physical operator

The operator uses genuine layer weights and deterministic Q8 activations on the M6000. Its control runs two stock MMVQ launches followed by the stock SwiGLU launch. The candidate uses one fused launch. Each fixture matched all **52,224 FP32 output words**, including a signed-zero case. Buffer guards and a post-timing replay also passed.

| Order | Stock group | Fused group | Latency saved | Paired wins |
| --- | ---: | ---: | ---: | ---: |
| ABBA | 484.755 µs | 466.104 µs | 3.85% | 8/8 |
| BAAB | 488.087 µs | 469.630 µs | 3.78% | 8/8 |

The short operator run fell between clock samples. The serving binary carries the same fused device instructions as the operator binary.

## Enable the measured route

Apply the [v13 serving profile](../dense-v13/README.md) with the v14 library from source commit `b2a821dcb6aa554afda8879b296d90de013f2146`. The measured CUDA library SHA-256 is `c63604052a92ee04213a97962c6e3dfbff7d9f839676ce60e44bf073e4691105`.

```bash
export GGML_CUDA_DENSE_N3_GLU=1
```

The opt-in matcher requires SM52, two adjacent exclusive matrix products from one shared FP32 activation, contiguous Q4_0 weights with 17,408 output rows and 5,120 input columns, three physical columns and unswapped bias-free SwiGLU. It also checks graph output ownership and memory ranges. Unmatched operations use the stock path. The default and value `0` select the stock dispatch.

## Two-slot follow-up

A separate OFF → ON run used the same v14 library and production two-slot MTP2 profile with frozen 1,024- and 2,048-token inputs. Each arm primed both slots with eight outputs, then ran a concurrent 64-output sentinel and two 192-output rounds per slot. All **16 requests and 1,824 generated token positions** matched, including text, cache, finish and MTP counters. The sentinel overlapped about 5 seconds and warm decode about 14 seconds. Both arms witnessed 81,803 N6 MMVQ host launches or graph captures. The ON arm also witnessed N3 fusion during solo priming.

| Warm round | OFF aggregate tok/s | ON aggregate tok/s | Goodput change |
| --- | ---: | ---: | ---: |
| First | 25.625 | 25.544 | −0.32% |
| Second | 25.588 | 25.560 | −0.11% |

Median warm goodput changed **−0.21%**. The second pair matched sampled P0 clocks at 1,088 MHz graphics and 3,304 MHz memory. The two-round, single-order run establishes output parity for the tested concurrent shape. [Two-slot receipt](two-slot.json)
