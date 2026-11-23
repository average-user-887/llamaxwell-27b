# Card 7: 27B C6 long-context prefill (23 Nov 2026)

| Measure | P/A: C6 lib, SMAX1=0 | B: SMAX1=1 | Delta |
|---|---|---|---|
| Proxy makespan (2 slots at 77.9K, each appends a 3782-token chunk and decodes 128 tokens), 8 runs per arm in 2x ABBA | 154.19 s (sd 0.94) | 152.82 s (sd 1.02) | **-0.89%** |
| Paired by sample position (smp0 / smp1) | 155.05 / 153.33 | 153.74 / 151.90 | -0.85% / -0.94% |
| Append prefill, sum of both slots | 67.57 t/s | 68.23 t/s | +0.98% |
| quick pp4k / pp23k | 90.50 / 83.94 | 90.70 / 84.12 | +0.2% / +0.2% |
| quick tg1 / tg2agg (1 sample) | 26.36 / 24.37 | 26.22 / 24.34 | -0.5% / -0.1% (noise; decode path unchanged) |
| quick probe outputs (3 prompts) | -- | identical | exact |
| test-backend-ops FLASH_ATTN_EXT (SMAX1 0/1, MINQ 1/64) | 3993/3993 | 3993/3993 | pass |
| FA bench, nk 92416, 2x512 q: softmax / full FA call | 74.8 / 479.6 ms | 60.2 / 465.3 ms | -20% / -3.0% |

Arm A is the new lib with SMAX1=0. That code path is unchanged, so it is identical to the C6 prod lib.

Verdict: the gain is real, consistent across both reps and far above the paired spread (~0.1%). At 0.89% it sits just under the >=1% gate.
- Not adopted on its own.
- The lever is bit-exact (same arithmetic, the mask is just read once per KV group), so it is a candidate for a batched-gains bundle.
- KLD was skipped because the speed gate did not pass, and the outputs are exact anyway.
- The full agentic2 run (~1h50) was skipped: no candidate passed, and it would not fit before the 15:30 GPU cut-off.

## Diagnosis: why the append prefill runs at "half speed"
- In agentic2 phase 2, both slots append at the same time, so the per-slot rates of 28-36 t/s add up to about 60-68 t/s. That is the tg_long single-slot rate. The halving is GPU sharing, not a slow kernel.
- At 92K depth, FA is about 45% of a prefill ubatch: 16 layers x 479 ms per 2x512 queries. Its two cuBLAS GEMMs run near peak:
  - QK 214 ms, PV 186 ms.
  - The default algorithm was best among all algorithms tried (-1, 0-23, 99, 100-115), within +-1%.
  - KC 2048 and merged non-batched GEMMs were slower.
- Exact-math headroom is therefore about 75 ms of softmax per call. SMAX1 recovers 15 ms of it.
- Larger levers, not exact or not kernel-side:
  1. Server scheduling. In Card 8, slot 1 starts about 41 s late, and decode tokens interleave with prefill ubatches; slot 0 decodes at 3.9 t/s while the other slot prefills.
  2. FP16 S/P at depth (S16): not exact, needs KLD.
  3. A fused FA-GEMM softmax that keeps S in shared memory.

Proxy note: the 2879-paragraph filler tokenised to 77.9K, not 92K. The relative FA share is slightly lower than at 92K, so the gain at 92K should be slightly higher.
Untested side patch for decode PV: fadec-pv-untested.patch (not built into this lib).
