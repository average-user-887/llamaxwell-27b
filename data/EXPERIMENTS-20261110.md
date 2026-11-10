# Experiments 2026-11-10 (M6000, sm_52)

All numbers measured on the Quadro M6000 24GB. KLD gate: one-sided d mean <= +.0010, same-top >= -.3 pt.

## Deployed (release prod-20261110-xmad)
- 27B (llamaxwell-27b 9a287a6): Q4_0 mmvq ncols 1-4 one-XMAD s16x2 dot (sm52-xmad.cuh), env GGML_CUDA_MMVQ_SM52=1.
  tg1 21.7 -> 26.35 tok/s. KLD bit-identical (0.010614, same-top 97.672). test-backend-ops q4_0/q8_0 pass.
- 35B (llamaxwell-35 81e5211): same dot, Q4_0 ncols 1-4 + MoE paths.
  2-slot aggregate ~78.5 -> ~83.7 tok/s; tg1 unchanged ~77.5. KLD 0.016649, same-top 94.38 (batch>1 only).

## Rejected / no gain
- Batched butterfly reduction (candidate B): no gain.
- fadec AV4 (2 tokens/thread, a1b1e70 on a worktree branch, not pushed): 3.194 vs 2.953 ms/call (AV3) at n_kv=100352; regs 248 vs 211. Phase A alone -15%, full kernel +8% slower.
- n=3 Q4_0 mmvq 4096x14336 (Exp1): 173 -> 170.7 us, target 150. Floor ~142 us memory + ~25-38 us issue per extra column.
  Ruled out: power/thermal (1151 MHz steady), bank conflicts, L1 sectors, wave tail. XPF prefetch (H14) helps n=3 only, hurts n=2/n=4. Needs load/compute overlap (warp-specialised or reg double-buffer). Not pursued.

## 35B decode profile (data: s35/prof35-out, s35/an.py, s35/run.sh)
Token 12.6 ms busy at d0, 13.7 ms at d16k (tg6 75.6 tok/s).
- occamy::moe_v2 40%/37%, 127 us/call, 125-142 GB/s
- gdn_fused 34%/31%, 141 us/call, 134-164 GB/s
- head_rowpack_sm52 12%/9%, ~240 GB/s (ok)
- mmvq attn q-proj 186 GB/s, o-proj 182 GB/s
- ofa_decode 3% -> 13% (d0 -> d16k), ~190 GB/s
Both moe_v2 and gdn_fused reaching 200 GB/s would save ~3.5 ms/token. In progress: moe_v2 bandwidth kernel behind an env flag.

## Prefill probe (data: pp/)
Baseline t/s, b2048: 27B ub1024 pp512 92.2, pp4k 95.5, pp16k 90.4. 35B ub512 pp512 458, pp4k 477, pp16k 397, pp64k 241.
- 27B pp4k: cuBLAS maxwell_sgemm_128x64_nn 84%, dequant 5.5%. Near Maxwell ceiling.
- 35B mul_mat_q q4_0 SASS: 128 regs, 41.5% XMAD + 19.5% BFE.
- SGEMM_NN / FA_GEMM on 35B: +0.
- 35B ub1024: 563 at 4k (+18%), 458 at 16k (+15%), no build. VRAM fit at prod args (-c 196608 -np 2) being verified.
- 27B ub1536 -0.9%/-0.5%; ub2048 +1.4% but does not fit 2x128k.

## Deck audition (deck-audition-table.md)
Prod stays deck-main. granite42-8b failed to load. qwen38-9b-distill worth a BFCL multi-turn rerun (35 vs prod 65). mellum21 is the only speed/quality trade worth considering.

## Layout
- ledger-27b/: full 27B optimisation ledger (LEDGER.md, ledger.jsonl, HANDOFF.md, per-experiment txt/py/sh). .bak files and binaries excluded.
- s35/: 35B profile, KLD and A/B logs from 2026-11-10.
- pp/: prefill probe scripts and results.
