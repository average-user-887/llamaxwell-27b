# Scout 3 pick: q4_0 mmvq on f32 activations (existing GGML_CUDA_FP32MV) on top of C4
Base: C4 (coordinator correction, 9 Nov; DEC=2, owner's decision). Design 2 itself is not reopened. No GPU used by this scout.

## Why (M = measured, E = my estimate)
- M: C4 is the failed-gate route. 64 chunks: mean .012500, max 6.679, top 97.641; the 16-chunk mean is .010614.
- M: removing q8_1 from that route (debug, cuBLAS, ~1 s/eval): 16-chunk mean .004487, max 1.641, top 98.309, 10:416 .387. Against .010614 that is -58%, the largest quality gain seen in this work. Runs differ in lib build and route, so not a clean pair.
- M: q8_1 is the amplifier: output noise 1e-6 gives l1 Q response 3.29e-3 with q8_1, 9.78e-7 without.
- M: a fast route already exists. `fp32mv_q4_0` in mmvq.cu takes f32 activations with no q8_1. Op n=3 214 -> 194 us (ledger). The flag string is in the C4 lib, whose md5 equals fadec2.so. No build for a first test. Its KLD has never been run (no ledger row, no use in f2/chain*).
- New evidence for re-testing (K-fp32mv was reverted 7 Nov, HANDOFF lists it as a trade-off): tg1 20.95 vs 22.49, accept 145/218 vs 153/203, tg2agg 21.26 vs 20.03, step 110.1 vs 110.5 ms. That run used one prompt (quick.py TG) and probe was .476, so the texts diverged. Accept was compared on different text. F2 now gives a mechanism for that divergence. Whether accept truly drops is open, not settled either way.

## Expected against C4 (all E, nothing below measured on C4)
- Quality at -b3: mean KLD -30 to -60%, >3 outlier at 10:416 gone. Solo decode only until the route gap below closes.
- tg@100k 14.39 and tg@52k 16.78: 0 to +3%. Short tg: -7% to +1%, the main risk. 2-slot aggregate: 0 to +6% (+6% is the 7 Nov single-text number).
- Chance it passes KLD and speed: about 50%. Honest size: mainly a quality gain plus a small speed gain, not another +78%.

## Known gaps (from source, M; effect unmeasured)
- Route: n=2..4 only, `!ids && !fusion`, rows %16 == 0. n=1 and n>=5 stay on q8_1, and 2 busy slots with drafts can reach n=6. A -b3 gate does not exercise these.
- Env: f2/kldrun.sh sets 3 GGML_CUDA flags plus test args; prod C4 sets 17 (NORM_Q8, Q8_REUSE, OCCAMY_*, GDN_COLS...). The 64-chunk gate therefore did not run on prod's env.
- NORM_Q8 + Q8_REUSE write a q8_1 copy that FP32MV never reads: wasted work at best. I did not read norm.cu far enough to exclude a stale f32 input. The first test uses prod env so a stale read would show as a KLD blow-up.
- Power: op -9% with step flat fits FP32 FMA drawing more power at the 250 W cap (hypothesis, E). nvidia-smi rows must log power and graphics clock.

## First test: KLD screen (no build, ~10 GPU min E: 64-chunk arm 1153-1160 s M, so ~290 s per 16-chunk arm)
Hold gpu-bench.lock. Stops prod for the run, then restore C4. f2/kldrun.sh with SO=prod-libs/libggml-cuda-fadec2-20261109.so, CHUNKS=16, base /w/head4.kld, `KLDSHIM_RAWTHR=99`, the full prod env from run-solo-27b-mtp-20261109-c4.sh.
- Arm C: C4 env. Arm F: C4 env + GGML_CUDA_FP32MV=1. Compare with f2/gate64.py.
Fixed now:
- Owner limits: F vs C d mean <= +.0005 and same-top no lower than -.3 pt.
- Mechanism: F mean <= C mean - .0030 (half the debug route's drop). Miss means coverage is short or the route does not cut the amplifier: stop, do not tune.
- Fail outright: nonfinite, or any F token KLD > 3 that C does not have.
Ruling needed: limits are worded two-sided (|d| <= .0005) and a drop this large breaks the wording while being an improvement. I read them as limits on getting worse. That reading is the owner's to confirm, not mine; get it before Stage 1 counts as a pass.

## Stage 1, only if the screen passes (no build, ~1 h GPU E)
- KLD 64 chunks, both arms on prod env, base head4-64.kld, same limits (~40 min).
- Speed on the same two arms: tg at 26 ctx over 8 distinct prompts (not TG + suffix), tg_long_fadec2.py at 51.9k and 100k twice each, tg2agg, pp4k; log accept, power, graphics clock.
- Keep: pooled short tg >= .97x C4, tg@52k >= 16.28, tg@100k >= 13.96, pp4k >= .97x, tg2agg no lower than -.8%, sustained clock >= .97x C4. Pooled accept reported.
- Short tg -3% to -7% with KLD pass: TRADEOFF row for the owner, not auto-kept.

## Stage 2, only if kept (build, Opus)
Extend fp32mv to n=1, 5, 6 so 2-slot prod leaves q8_1 entirely. Kernel register and launch tuning on sm_52, hence Opus. Then skip NORM_Q8 work when the route is active. Re-run Stage 1. Source only in a copied tree; lib to mmvqtune-libs (Storage 18 GB free).

## Runner-up: draft 3 at long context on C4 (no build, ~1 h)
Scout 2's unrun runner-up. E: +10% filler @100k, about -2% real text, -10% short (older base). n=4 op 274.6 vs n=3 214.2 us (M), and DRAFT is a fixed server flag, so a short-context loss hits all work unless draft length becomes context-dependent (server C++). Lower ranked: expected gain is on filler text only.

## Fallback if FP32MV's accept loss proves real
16-bit integer activations in the mmvq path (bit width fits int32: 4+16+5 bits). Finer steps, keeps integer numerics. Build, Opus. Gain unmeasured; thresholds remain, so amplifier shrinks rather than vanishes (E).

## Rejected
- SPLIT=1/2 "passes" (16-chunk): lottery, SPLIT=3 + noise gives .334 and 6.300 at the same token; also design 2 work.
- Any further decode FA work: excluded by brief.
- Checkpoint Stage 2: Stage 1 no gain, on hold, no new evidence.
- Native prefill FA: E <= 8-10% pp@100k for heavy work, sgemm is 79% of prefill.
- Higher clocks, memory overclock: power capped at 250 W, memory temp unreadable, ECC off.
- Two-plane q8 activation: doubles n, E +60-85% mmvq time.
- mmvq padding/re-split: n sweep closed it. sm52 XMAD: op -3% vs 15% gate.
- HANDOFF rejected list (ub2048, draft KV q8, IQ4_XS, Q3_K_XL...): no new evidence.
