# Card 1: C5 rebaseline on M6000 (22-23 Nov 2026)

| Metric | Session A (22 Nov 22:07) | Session B (23 Nov 01:42, after restarts) | Delta B/A |
|---|---|---|---|
| pp4k (t/s) | 91.3 | 90.4 | -1.0% |
| pp23k (t/s) | 83.88 | 83.75 | -0.2% |
| tg1 (t/s, max of 2) | 26.24 | 26.31 | +0.3% |
| tg2agg (t/s) | 21.99 | 22.00 | 0.0% |
| accept (MTP) | 154/200 | 154/200 | - |
| probe vs old base | n/a (first) | 1.0 (vs A) | - |
| vram peak (MiB) | 23218 | 23218 | - |
| tg_long tg @22.8k tokens | 22.65 | 22.68 | +0.1% |
| tg_long tg @51.9k | 19.35 | 19.37 | +0.1% |
| tg_long tg @100.2k | 16.12 | 16.11 | 0.0% |
| tg_long pp @22.8k/51.9k/100.2k | 83.91/75.21/64.50 | 83.82/75.36/64.55 | <=0.2% |
| agentic2 makespan mean (3 reps) | 1384.7 s (1383.9/1384.4/1385.7) | 1517.6 s (1511.4/1521.2/1520.1) | +9.6% |

Each quick23 value is the mean of 2 reps (QUICK_LONG=1). The old base_m (pre-xmad fagemm lib) was tg1 22.66, tg2agg 21.39, pp4k 91.3, pp23k 84.79; C5 is tg1 +16%, tg2agg +2.8%.

## agentic2: INVALID as a makespan metric at this size (needs a fixture fix, not run again)
- filler_reps=3109 (calibrated by /tokenize to ~90K) gave 96.8K (A) / 99.9K (B) tokens per slot, i.e. over the 84-96K window. Use about 2800.
- Phase 1 (both slots prefill ~97K concurrently, n_predict 1) hit the 1800 s HTTP timeout in `q.post` (default). Only slot 0 finished (1543 s / 1637 s). Slot 1 had only ~21-26K cached.
- So in phase 2 slot 1 re-prefilled ~75-83K tokens while slot 0 stalled (tg 0.09-0.11 t/s). The makespan (~1385 s / ~1518 s) mostly measures that prefill, not decode. The A/B +9.6% gap comes from the 3% larger prompt (99.9K vs 96.8K) plus the same stall, so not a clean noise estimate.
- Fix needed (code, not done): raise the `go()` request timeout above 1800 s (or prefill the slots one at a time) and use filler_reps about 2800. Prefill alone is about 25-27 min per slot at 62-63 t/s.
- Noise on the other metrics is <=1% (pp4k) and <=0.3% elsewhere.

## Step 4: which mmvq path the two-slot MTP verify takes (no code changed)
- Source (moe-v2-20261114/src/ggml/src/ggml-cuda/mmvq.cu, not the exact xmad27 commit 9a287a6): the sm52 XMAD Q4_0 dot is guarded by `ncols_dst <= 4` (lines ~917, ~1304). The generic mmvq kernel handles ncols up to MMVQ_MAX_BATCH_SIZE (8).
- Runtime: ran the C5 config with `-lv 4` and two concurrent 96-token decodes (verb.py, verb-docker.log). The llama-server log (this build) prints no per-decode batch n_tokens, so the batch size was NOT directly observed. Observed: each slot ran about 12.0 t/s concurrently versus about 22.7 solo (tg2agg 22.0 about equal to tg1 22.7 with this prompt), MTP draft 2, accept about 75%, one `graphs reused` stream.
- Inference: with np2 and draft 2, a concurrent verify is 2 slots x 3 tokens = 6 rows in one batch, which is over the XMAD cap of 4. The most likely path is the generic (non-XMAD) mmvq at ncols=6. This is consistent with tg2agg (22.0) being far below 2x tg1 (26.3). Not proven. A proof needs a counter in mmvq launch (ncols_dst, sm52 flag) or ggml-cuda debug counters; a decode-batch log line in llama-server would also settle it. Suggest a small patcher card.
- Note: whether the 6-row batch is split into two 3-row calls could not be observed. No source evidence that it is.

## Setup
- Base edit: state.json `base` backed up as state.json.bak-20261122-base; old base and base_m kept in `notes.old_base_20261122`. quick.py backed up as quick.py.bak-20261122-base. quick.py change: DEADLINE constant 2026-11-13 -> 2026-11-30 (stale deadline made run_case skip), nothing else. state.json base now: so=prod-libs/libggml-cuda-xmad27-20261110.so, env = 4 base vars (MMVQ_SM52, FA_DEC=2, SGEMM_NN, FA_GEMM) + the other 14 in quick.py ENV = the C5 script's 18 vars. model Qwen3.8-27B-Q4_0-head4.gguf, np2, c 262144 (2x131072), q4_0 KV, b2048, ub1024, draft-mtp n=2, -no-kvu, --cache-ram 4096, ctx-checkpoints 8.
- lib: libggml-cuda-xmad27-20261110.so sha256 8ac6aa42fb1960a57028f77a8072ee3205cb60cdffb59147c52b0981a395522a
- model: 15388072864 bytes, mtime 2026-11-07 16:01:28 (no sha256 computed)
- image sha256:3a0caa9afea12f5ca5d982621cc2d86b1fa6ac614d735b262fa475d7dc6e2a6d
- GPU (temps.csv, 10 s samples over the whole chain, incl. idle): temp 42-74 C (limit 80, never reached), sm clock up to 1151 MHz under load, power up to 290 W peak draw (cap 250 W is a sustained limit).
- Method: chain.sh under `flock gpu-bench.lock`, detached (setsid), NO_RESTORE guard so prod stayed stopped, exit trap removes NO_RESTORE and the exp container. Per session: quick.py 23 (QUICK_LONG=1) -> tg_long.py 1200 2600 4900 -> agentic2.py 3 3109. A and B are separate server starts (each script restarts the container; ~3 h apart, 2 starts of the container between).
- Caveats: ledger has one stray NOFIT/FAIL BASE-rep0 row from my first aborted launch (22:07, killed); LEDGER.md/ledger.jsonl updated by quick.py. Prod llamacpp-m6000-27b left stopped, untouched.
- Files (this dir): chain.sh, calib.py, verb.py, chain.log, calib.log, quick23-{A,B}.log, tglong-{A,B}.log, agentic2-{A,B}.log, temps.csv, verb-docker.log
