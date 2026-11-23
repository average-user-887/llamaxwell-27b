# 27B card 2: XMAD Q4_0 mmvq for ncols 5-8 (C6), 23 Nov 2026
Finding: Qwen3.8-27B with MTP draft 2 and 2 slots verifies 6 columns per step (ncols=6: 32.9k launches, ncols=8: 456, ncols=3: 161k in a real 2-slot run).
The C5 XMAD Q4_0 path was guarded by ncols<=4, so ncols 5-8 used the generic mmvq. Guard widened to <=8 (env GGML_CUDA_MMVQ_SM52_N8, default on).

Result (v2 lib, sha256 38ae66b9df7a282bdbbbf9af310fd665ecee71e90ffe3d890595bec07bbe0062), ABBA on one lib, N8 off/on/on/off:
| arm | tg1 | tg2agg | pp4k |
|---|---|---|---|
| off | 26.35 | 22.07 | 91.6 |
| on | 26.34 | 24.37 | 91.0 |
| on | 26.34 | 24.13 | 90.9 |
| off | 26.27 | 22.17 | 89.9 |
tg2agg +9.6% (22.12 -> 24.25); tg1, pp4k flat; probe 1.0, draft accept 154/200 in every arm. Release gate >=1%: crossed.

Quality: test-backend-ops MUL_MAT q4_0 CUDA vs CPU 118/118 pass (N8 on and off). KLD at -b 6 (needed to reach ncols 6; the standard b3 gate cannot), 16 chunks vs head4.kld: xmad27 and v2/N8=1 per-token outputs byte-identical (mean KLD 0.008942, same-top 97.623%); gate64.py crashed on zero variance, so no printed GATE verdict, the runs are bit-identical.
Not run: standard b3 64-chunk gate (ncols 1-4 code unchanged), tg_long, pp23k, agentic2 (fixed fixture untested).

Correction (additive): v1 lib (5d76c140...) is invalid: tree carried commit 602a131 (debug mmvq_tune.h, not in deployed libs), giving tg1 ~21.9. Fixed by d63ca69 restoring the deployed tune values; v2 reproduces xmad27 exactly with N8=0.
Prod: run-solo-27b-mtp-20261123-c6.sh (rollback in its header). Baseline: ../card1-baseline-20261122.
