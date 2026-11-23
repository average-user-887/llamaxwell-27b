# Card 8 (23 Nov 2026): agentic2 two-slot baseline on C6 prod lib

| Metric | Rep 0 | Rep 1 | Mean |
|---|---|---|---|
| Valid (cache_n >= 90% of prefill) | true | true | - |
| Prompt per slot (prompt_n) | 92320 | 92320 | - |
| Prefill phase wall, 2 slots sequential (not in makespan) | 2795.4 s | 2802.4 s | 2798.9 s |
| Prefill pp per slot | 66.2 / 66.1 t/s | 65.9 / 66.0 t/s | ~66 t/s |
| Makespan (timed phase: +3873 tok tool chunk, 128 tok decode, both slots parallel) | 176.47 s | 174.71 s | 175.59 s |
| of which tool-chunk prefill (pp 28.2 slot0 / 32.2 slot1 t/s) | ~137 / ~120 s | same | ~78% of makespan |
| of which decode (128 tok at tg below) | ~37 s slot0 / ~15 s slot1 | same | ~22% |
| tg per slot at ~96K ctx, both slots active (slot0 / slot1) | 3.44 / 8.54 t/s | 3.45 / 8.40 t/s | 3.45 / 8.47 |
| MTP accept (slot0, slot1) | 77/100, 79/96 | 77/100, 78/96 | ~79-80% |

Baseline makespan = 175.6 s (min 174.7). Run-to-run spread 1%.

## Method
- Lib prod-libs/libggml-cuda-xmadn8-v2-20261123.so, state.json base (np2 -no-kvu c262144 q4_0 KV, b2048 ub1024, draftn 2, head4 model, env as run-solo-27b-mtp-20261123-c6.sh), experiment container llamacpp-m6000-27b-exp port 11448 via quick.py start().
- `python3 agentic2.py 2` (calibration fix from card 5 worked: filler_reps 2879, 30.73 tok/rep, no HTTP 400) run by run.sh under flock gpu-bench.lock, wrapped in labexp run; llama-swap unloaded by POST /api/models/unload, prod docker-stopped, restored at end (prod healthy on 11437).
- Run began 09:35 (lock free immediately); prod restored by script at the end; Card 7 did not hold the lock at the time.
- Temps sampled every 10 s (temps.csv): max under 80 C hard limit.

## Caveats
- Total wall ~1 h 50 min (2 x 2800 s prefill + makespans), over the 90 min cap because 2 reps of 2x92K prefill at 66 t/s cannot fit; the run had already finished before the follow-up message (13:20), so no rerun was done.
- Makespan is dominated by tool-chunk prefill at deep context (pp drops to ~28-32 t/s at 92K depth with both slots), not decode. per-slot tg is uneven (3.4 vs 8.5 t/s), cause not investigated; compare runs by makespan.
- No other session was using the GPU during the run.

## Files
card8-20261123/{run.sh,run.log,outer.log,agentic2.log,temps.csv,RESULT.md}
