# M6000 Qwen3.8-27B speed work: handoff (7 Nov 2026, evening)

## Where things stand (all measured, M6000 sm_52, 2 slots × 128k, q4_0 KV)
| metric | prod C1 | best now (C3) | gain |
|---|---|---|---|
| pp 4k t/s | 72.3 | 90.8 | +26% |
| pp 22.7k / 51.9k / 100k | 78.0 / 65.5 / 51.7 (C2-class) | 83.7 / 75.4 / 64.3 | +7 / +15 / +24% |
| tg1 (MTP draft 2) | 16.2 | ~22.4 | +38% |
| tg 2 slots aggregate | – | ~21.5 | |
| quality | | head4 KLD 0.0022, top 98.6%; all kernel changes KLD ~0 | |

R21 (FA-gemm adopt gate, pp4k/tg1 ≥0.97× base): KEEP — pp4k 90.8 (vs 89.2), tg1 22.32 (vs 22.35), tg2agg 21.41, probe 1.0, VRAM 23.2 GB.

## The pieces that make up C3
`run-solo-27b-mtp-20261107-c3.sh`, with C2 and C1 kept as rollbacks:
- **Model and layout:** head4 Q4_0 GGUF; -np 2 -c 262144 -no-kvu; q4_0 KV; ub 1024; MTP draft 2.
- **Library:** `prod-libs/libggml-cuda-fagemm-20261107.so`, built by `build_b11552_fagemm.sh` from `src-b11552-mmvqtune-20261107`. It contains:
  - the mmvq p4n1 decode tune;
  - SGEMM_NN dqT2 (prefill dense GEMM);
  - `ggml-cuda/fattn-gemm.cu`, which runs prefill FA as cuBLAS GEMMs. It only takes Q ≥ 64 rows, so decode and drafts are untouched.
- **Env:** prod env plus `GGML_CUDA_SGEMM_NN=1 GGML_CUDA_FA_GEMM=1`.
- **Adopting C3 is the owner's call:** `docker stop llamacpp-m6000-27b && docker rename llamacpp-m6000-27b llamacpp-m6000-27b-c1 && bash run-solo-27b-mtp-20261107-c3.sh && labslot check`.

## How to test (keep it this simple)
- **Tools you need:**
  - `opt27b-20261113/quick.py --round N`: one case per lever. It starts the exp container, measures pp4k, tg1 and tg2agg plus probes, writes a row to LEDGER.md, and gates it with `gate()` (7 Nov). Noise is about 0.2%, so a metric counts as moved at 3σ (0.6% for tg1/pp4k/pp23k, 0.8% for tg2agg). KEEP means some metric ≥ +3σ and none ≤ −3σ. CONFIRM (+1.5–3σ) re-runs automatically and decides on the mean; if it is still small but consistent it goes to SHELF. TRADEOFF (one metric ≥ +2% while another drops) goes to `state.json` `tradeoffs` with its full config for the owner. SHELF and TRADEOFF ideas are saved, never lost. With `QUICK_LONG=1` every case also measures pp at 22.7k (pp23k); run `quick.py 23` (rebase, 2 reps) first so the base has it. Add a round as a 10-line `roundN(s)` function.
  - `fa_eval.py <so> <reps…>` with `VARS="0 1"`: long-context pp A/B. reps 1200/2600/4900 give 22.7k/51.9k/100k tokens.
  - KLD gate for numerics-only changes: see the perplexity block in `fa_eval.sh`, base `/media/avg-usr/Models/kld-head4-20261107/head4.kld`. Pass means mean KLD ≈ 0 and same-top ≥ 99.5%.
  - Kernel correctness: `test-backend-ops -o <OP>` from `src-…/build/bin`.
  - Profile: `/media/avg-usr/Models/prof-27b-20261107/prof_pp.sh` (nvprof GPU summary).
- **Rules:**
  - Always hold `flock ../gpu-bench.lock`.
  - Don't restore prod between runs (owner, 7 Nov); restore at session end with `docker start llamacpp-m6000-27b` and `labslot check`.
  - Don't run `docker rm` on prod.
  - Back up every edited file as `.bak-YYYYMMDD-tag`.
  - The Storage disk is 98% full (19 GB free); delete nothing, and put new libs in `mmvqtune-libs/`.

## Remaining tuning points across the chain (biggest first, by my estimate)
1. **Long-context pp:** it still drops from 89 at 4k to 64 at 100k.
   - The FA-gemm S scratch is fp32; switching it to f16 S, or fusing the mask and softmax into the GEMM epilogue, halves its traffic.
   - Profile at 50k to see whether FA or GDN chunk scan dominates now.
2. **tg at long context:** never measured (every tg number is at short context).
   - Add a tg1-at-50k/100k row.
   - The decode FA (tile/vec at q4_0 KV) is likely the drop; a split-K vec kernel tune is the lever.
3. **Short pp:** ~89 t/s is about 4.8 TF; cuBLAS on Maxwell tops out near there.
   - Remaining levers: a fused dequant + GEMM to avoid the f32 round trip, ub 1536, and GDN prefill kernel time (profile share first).
4. **tg1 at 22.4, memory-bound:**
   - MTP accept rate: draft 2 is the best fixed setting; an adaptive draft length is untested.
   - Small-op fusion and launch overhead: check graph replay gaps in nsys.
5. **Clocks (owner, needs sudo):** apps clocks are 987 MHz while the board supports up to 1152.
   - Try `sudo nvidia-smi -i GPU-684f5e72-ce85-7b4f-e4f6-2dddddd6e60e -ac 3305,1152`, then A/B pp4k and tg1.
   - The board may hit its 250 W cap; revert with `-rac`.
6. **Trade-offs for the owner, not auto-kept:**
   - K-fp32mv: tg2agg +6% but tg1 −7%.
   - draft1: tg2agg +2.6% but tg1 −7.7%.
   - ckpt0: no gain, and it loses prompt-cache reuse.

## Already rejected (don't retry)
ub2048, draft 3/4/6, draft KV q8, graphs off, rms_warp1, the mmq/occ/q8reuse/rowpack toggles, IQ4_XS, Q3_K_XL, nnsplit, mmvq RPB8/xcols/n3 variants, FA_GEMM_KC below 4096. Vision mmproj does not fit on 2×128k.

Files: `LEDGER.md` (every row), `MORNING.md` (dated summaries), `state.json` (quick.py base), `fa-gemm-*.txt`.
- 7 Nov ~21:50: C3 ADOPTED into prod (llamacpp-m6000-27b, FA_GEMM=1, head4); old C1 container kept as llamacpp-m6000-27b-c1-20261107; registry updated (bak serving-registry.json.bak-20261107-c3); labslot all OK.

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
