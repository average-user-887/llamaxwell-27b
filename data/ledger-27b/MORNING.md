# opt27b morning summary (night run 06-07 Nov, all 4 rounds rc=0, prod healthy on :11437)
- Layout: 2 slots x 131072 ctx each, KV q4_0 (F1). VRAM 23.4 GB. 2x262144 unified (F2) also fits (23.7 GB), same speed. 2x128k with q8_0 KV does NOT fit.
- Best config vs prod: q4_0 KV (prod q8_0, 1 slot), 2 slots, ub1024 (prod 512), MTP draft 2 (M2). No env flags helped.
- Speed (pp4k / tg1 / tg2agg):
  - control P (prod, 1x128k q8): 72.3 / 16.18 / -
  - best (F1+ub1024+draft2): 78.5 / 17.4 / 17.6
  - old baseline: 73.2 / 19.0 (MTP). Control today is below the old tg; best is still below old 19.0.
- tg >= 21 target: NOT hit (best 17.4 t/s).
- Vision mmproj: does not fit (24188 MiB peak on the 2-slot layout, run flagged VISION-NOFIT).
- Rejected: ub2048 (error), draft4/draft6 (error, tg collapse), draft KV q8 (-0.3%), graphs off, rms_warp1 (probe 0.667 quality drop), all mmq/occ/q8reuse/rowpack env toggles (within +-0.3%, noise).
- Quant candidates (round 4): IQ4_XS 22.3 GB, tg 14.4, probe 0.40; Q3_K_XL 21.2 GB, tg 13.8, probe 0.53. Both slower and lower quality than current, not worth switching.
- Noise (B-rep x3): tg1 17.39-17.42, pp4k 78.0-78.5. Noise is about 0.2%, so only gains >1% are real; only ub1024 (+9.9% pp) and draft2 (+5% tg) are.
- Next lever: tg is memory-bound at ~17; the old 19.0 MTP run was faster, so find why (build/flags/MTP version difference vs. the old baseline) before trying new knobs. Alternative: draft 3 (untested) or a bigger-bandwidth quant is not available.

## 7 Nov 17:15 — day loop result (rounds 11–14)
- Best: head4 Q4_0 + mmvq p4n1 .so + 2x128k q4_0 KV + ub1024 + MTP draft2 → tg1 22.4, tg2agg 20.0, pp4k 79, VRAM 23.2 GB (3 reps, stable).
- vs prod C1 (1x128k q8, tg1 ~19): +18% tg, 2 slots instead of 1.
- head4 quality vs head6: PPL +0.07% (noise), mean KLD 0.0022, same-top 98.6% (kld-head4-20261107.txt).
- Reverted: draft1/3/4, mmvq RPB8, xcols, n=3 nwarps 1/4 (n=3 kernel stuck ~217 us).
- Adopt (owner): docker stop llamacpp-m6000-27b && docker rename it to llamacpp-m6000-27b-c1 (or rm), bash ../run-solo-27b-mtp-20261107-c2.sh, labslot check. Rollback: run-solo-27b-mtp-20261107-c1.sh.
- Prod restored to C1 at 17:12, labslot all OK.

## 7 Nov evening, rounds 15-20 (current best, not in prod)
- Best: F1 2x128k q4 KV, ub1024, draft2, mmvq-p4n1, head4, SGEMM_NN=1, dqT2 lib (mmvqtune-libs/libggml-cuda-mmvqT-sgemmnn2.so).
- vs prod P (1x128k): pp4k 72.3 -> ~89.1 (+23%), tg1 16.18 -> ~22.3 (+38%), tg2agg ~21.45; pp at 37k 71.0 (baseline 65 at 32k).
- KLD (not probe) gates numerics-only changes: sgemmnn/dqT2 mean KLD ~0, same-top 100%.
- Rejected: nnsplit (-2%), ckpt0 re-test (+0.4% tg1, +2.8% pp ~noise; loses hybrid prompt-cache reuse).
- Owner trade-offs: K-fp32mv tg2agg +6%/tg1 -7%; draft1 tg2agg +2.6%/tg1 -7.7%.
- Next lever: long-context FA/GDN (pp drops 18% from 4k to 37k); needs kernel work.

## 7 Nov night, rounds 21 (FA kernel long shot), current best = C3
- New: GEMM prefill FA (ggml-cuda/fattn-gemm.cu, GGML_CUDA_FA_GEMM=1); tbo 3993/3993, KLD 0 (max 5.3e-5, top 100%).
- pp 22.7k/51.9k/100.2k: 78.0/65.5/51.7 -> 83.7/75.4/64.3 (+7/+15/+24%); pp4k 90.8; tg1 22.32; tg2agg 21.41.
- vs prod C1: pp4k +26%, tg1 +38%, 2 slots. Adopt script: ../run-solo-27b-mtp-20261107-c3.sh (owner call). Clock raise needs sudo.
- Handoff for next agent: HANDOFF.md.
- 7 Nov ~21:50: C3 ADOPTED into prod (llamacpp-m6000-27b, FA_GEMM=1, head4); old C1 container kept as llamacpp-m6000-27b-c1-20261107; registry updated (bak serving-registry.json.bak-20261107-c3); labslot all OK.

## 7 Nov 22:08-22:53, round 22: app clocks 987 -> 1152 MHz (owner sudo, 78C guard)
- Power-capped: throttle reason SW power cap (0x4) on 1313/1314 samples; loaded SM median 1037 MHz (p10 999, p90 1113); peak 76C, 287 W spikes.
- pp4k 91.2 vs 90.8 (+0.4%), tg1 22.26 vs 22.32 (-0.3%), tg2agg 21.46 vs 21.41: noise, revert by quick.py rule.
- long pp 22.7k/51.9k/100k: 85.5/76.4/65.3 vs 83.7/75.4/64.3 (+2.2/+1.4/+1.5%), A = 20:01 run at 987.
- Verdict: about +1.5% long-ctx only, for 2C below the guard and 74C+ for 86% of loaded time. Not worth it; power, not clock, is the limit (max power limit 250 W).

## 8 Nov 00:xx, chain1 (same-session clock sweep + long-ctx tg)
- 987 vs 1152, rebase 2 reps each: pp4k 91.3/91.3, tg1 22.72/22.66, tg2agg 21.40/21.39, pp23k 83.98/84.79 (+0.96%). Peak 74/75 °C. Pick 1152 (rule: pp23k ≥ +0.6%, no metric < −0.6%, peak ≤ 77). Now set at 3305,1152. state.json = 1152 base.
- tg at long ctx (1152, C3, n_predict 128): 26 tok 18.9 t/s; 51.9k 11.58; 100k 8.07 (accept 79/95, 80/94). Decode is 2.3× slower at 100k, so decode FA is the top lever: Card D (GGML_CUDA_FA_DEC, separate tree src-b11552-fadec-20261108).
- Queued overnight: Card A nvprof 50k, Card B FA-gemm f16 S, Card C sm52 mmvq (scout pick), Card D decode FA.

## Night 7→8 Nov results (orchestrator; details in LEDGER.md "8 Nov overnight" + card rows)
- Clock: 1152 MHz KEEP (pp23k +0.96%, rest noise, peak 75 °C; power-capped). Apps clocks now 3305,1152; revert `sudo nvidia-smi -i <uuid> -rac`. Prod = C3, unchanged, restored 05:5x, labslot all OK.
- Long-ctx decode is the big gap: tg 19.1 / 11.6 / 8.07 t/s at 26 / 51.9k / 100k (draft 2). Draft 1: 19.1 / 8.09 / 5.20; MTP off: 14.2 / 5.80 / 3.73. Keep draft 2.
- nvprof 22.7k (prof-c3-50k-summary.txt): prefill sgemm 79% (ceiling), FA 9.6%; decode mmvq 58.8%, flash_attn_tile 22.5%, q4_0→f16 KV dequant 11.3%, GDN 2.8%.
- Rejected tonight: FA-gemm f16 S (noise; fagemm-s16.so), sm52 XMAD mmvq (op −3%, gate 15%; fagemm-sm52.so), decode Q3-4→VEC (tg@100k −37%; fagemm-fadec.so; VEC re-reads KV per q-head).
- Next lever (biggest, unbuilt): GQA-grouped q4_0 decode FA (6 q-heads × ≤4 rows per kv-head, KV read once, dequant in registers, split-K). Spec: fadec-design2-spec.md; tree src-b11552-fadec-20261108 (flag GGML_CUDA_FA_DEC). KLD gate at -b3 needs a flag-0 -b3 control (baseline numerics differ from c512).

## 8 Nov Card F2: design 2 KLD token (DEC=2 not adopted, owner decides)
- Token: chunk 10 pos 416 ".catch". Base .397, DEC=0 .294 (KLD .064), DEC=2 .99995 (KLD 6.679). Finite, same top token, over-confident.
- Verdict: rounding-level differences amplified through q8_1 activation rounding; also reachable on stock (stock + 1e-7 output noise reaches 4.50 at the same token). No DEC=2 kernel bug shown, no fix built.
- 64 chunks: DEC=0 mean .011873, DEC=2 .012500, paired +.000627, t 1.11. >3 tokens: 0 vs 1. Gate 2 still FAILS as written on the 16-chunk test (+.00152); not relaxed.
- Not determined: 10:416 without q8_1 (run reached chunk 0 only), true >3 rate. No-q8 arm estimate 25-50 min per arm.
- Files: fadec2-kld-token-20261108.txt; LEDGER "8 Nov Card F2"; lib mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so unchanged; 64-chunk base and debug v3 lib in /media/avg-usr/Models/kld-f2-20261108.
- Prod restored to C3 after the card. NO_RESTORE renamed NO_RESTORE.done-20261108b.

## 9 Nov no-q8 test and 64-chunk re-gate (design 2 NOT adopted)
Limits fixed before results: |dmean| <= .0005, same-top within .3 pt. Raw: fadec2-noq8-regate-20261109.txt; tools f2/chain9.sh, f2/gate64.py.
| Run | n | stock mean | DEC=2 mean | d | t | top d | max | 10:416 | gate |
|---|---|---|---|---|---|---|---|---|---|
| no-q8 (v3 debug lib, 16 chunks; `--chunks 11` was ignored, header "0-10" is wrong) | 4080 | .004384 | .004487 | +.000103 | .76 | -.074 | 1.676 / 1.641 | .062 / .387 | PASS |
| q8_1, real fadec2.so, 64 chunks | 16320 | .011873 | .012500 | +.000627 | 1.11 | -.153 | 1.498 / 6.679 | .064 / 6.679 | FAIL |
- Real lib vs F2 debug lib per-token differences are exactly zero (gate64.py hit se=0 on that check, the traceback in the raw file).
- Outlier absent without q8_1, which supports the amplifier reading; 10:416 stays about 6x higher on DEC=2, so not identical to stock.
- Chain bug: KRUNMAX sleep held the tee pipe; GPU idle and prod down about 30 + 13 + 37 min. Fixed in kldrun.sh (backup .bak-20261109-pipe). No results affected.
- Prod back on C3 at 11:17.

## 9 Nov C4 ADOPTED: design 2 in prod (owner's decision, gate still FAIL)
- Owner: "adopt design 2, the gains are worth it". This supersedes the "NOT adopted" outcome in the section above. The gate result itself stands: 64 chunks FAIL (+.000627 vs limit .0005, top -.153).
- Gains: tg@100k 8.07→14.39 (+78%), tg@52k 11.59→16.78 (+45%); short tg -0.39%, pp -0.02 to -0.15%.
- Known cost: one token at chunk 10 pos 416 (.catch) goes KLD .064→6.679, same top token, deterministic.
- Prod: `llamacpp-m6000-27b` from `../run-solo-27b-mtp-20261109-c4.sh` (C3 + lib `prod-libs/libggml-cuda-fadec2-20261109.so` + `GGML_CUDA_FA_DEC=2`). C3 container kept as `llamacpp-m6000-27b-c3-20261109`.
- Rollback: `docker stop llamacpp-m6000-27b; docker rename llamacpp-m6000-27b llamacpp-m6000-27b-c4-off; docker rename llamacpp-m6000-27b-c3-20261109 llamacpp-m6000-27b; docker start llamacpp-m6000-27b`
- Later GPU work restores prod to C4, not C3. Live check: `c4-prodcheck-20261109.txt`.

## 9 Nov FP32MV screen on the full prod env (16 chunks, `fp32mv-screen-20261109.txt`)
- C4 vs C4 + `GGML_CUDA_FP32MV=1`: mean .010614→.004376 (t -3.75), same-top 97.672→98.480, max 6.679→1.732, token 10:416 6.679→.034. Pre-fixed screen limits all met. gate64.py prints FAIL only because its check is two-sided; owner ruling needed on that wording.
- Prod-env C3 route (DEC=0) .009094/1.498079 and C4 .010614/6.679251 match the 3-flag runs, so the extra prod flags did not change these numbers. The DEC=0 repeat also matches exactly when heat-soaked.
- Speed not measured yet. Not adopted.

## 9 Nov KLD gate limit raised (owner)
- Owner: "increase treshold for kld, i'll test the model myself later". Value chosen by Claude: d mean <= +.0010 (was |d| <= .0005). Same-top limit stays .3 pt. Both now one-sided, so improvements cannot fail. In `f2/gate64.py`; backup `.bak-20261109-kldlimit`.
- Owner also authorised automatic merge to prod when gates pass, repeating the cycle, while away.

## 9 Nov FP32MV Stage 1: no auto-merge, TRADEOFF for owner
C4 stays prod. Results: fp32mv-stage1-20261109.txt/.jsonl. C4 -> C4+FP32MV.
- KLD 64 chunks PASS: mean .012500 -> .005521, top 97.641 -> 98.536, max 6.679 -> 2.605, 10:416 6.679 -> .034.
- Short pooled tg 22.46 -> 22.95 (1.022x), tg1 21.70 -> 21.41, tg2agg 21.79 -> 22.07, pp4k 91.85 -> 91.60, clock 1111 -> 1141 MHz. Accept 624/774 -> 624/775.
- tg@52k 17.80 -> 16.91 (.950x), passes fixed limit 16.28. tg@100k 13.41 -> 13.75 (1.025x), FAILS fixed limit 13.96.
- Limits were not changed after results. Both long limits came from earlier C4 runs (16.78, 14.39) that do not match same-run C4. Long rows use one text; accept differs by arm (52k 80/94 vs 77/99), so the -5% and +2.5% may be text effects. Not resolved.
- Next if wanted: long-context tg over several prompts; Stage 2 build (n=1,5,6) waits for owner's call or a clean pass.

## 9 Nov: power cap and llama-swap
Power sweep stopped early, owner keeps 250 W; table in LEDGER.md. 225 W costs 4-5% for up to 6% less energy.
27B and 35B now run on call behind the host llama-swap (units m6000-llama-swap, m6000-swap-lan-forward; config llama-swap/host.yaml). A swap takes about 35-50 s. During benches hold gpu-bench.lock: swap then refuses loads. Stopping the service also stops the loaded model.
