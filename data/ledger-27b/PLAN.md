# Plan: quick runtime gains for Qwen 27B on the M6000 (tonight, 6→7 Nov)

## Context
The owner reset the plan on 6 Nov.
- **Goal:** the cheapest runtime gains for dense Qwen 27B on the M6000 (sm_52, 24 GB), using the Occamy method of short keep-or-revert rows with a ledger.
- **Session limits:** keep every M6000 session short, and don't build an elaborate harness.
- **Targets:**
  - tg (MTP) up from 19.0 to at least 21 t/s, roughly +10%. That is good enough to start the local improvement loop.
  - 2 slots at 128k each is good, 96k is okay, and 2×64k is the minimum.
  - Vision head (mmproj) only if it fits. Otherwise a small vision model on the Deck or the 1660 Ti, not tonight.
- **Baseline:** taken from existing aggregates (labbench measure-20261107.json): pp 73/65/47 at 4k/32k/120k, tg 11.8 without MTP and 19.0 with MTP, 19.4 GiB at 1×128k q8.
- Scrapped from the old plan: the 120k needle runs, the front proxy, and kernel work.

## Fit maths
- **Weights:** about 14.6 GiB (Q4_0-head6), with MTP, plus about 0.6 GiB of compute buffers.
- **KV:** only 16 of the 64 layers hold KV, about 34 KB per token at q8.
- **Options:**

| Layout | KV | Est. total | Verdict |
|---|---|---|---|
| 2×128k at q8 | about 8.5 GiB | about 23.7 GiB | too tight |
| 2×128k at q4_0, or 256k unified | about 4.5 GiB | about 21.5 GiB | fits, with room for the mmproj (about 0.9 GiB) |
| 2×96k at q8 | about 6.4 GiB | about 21.6 GiB | fits |

These are only estimates; the short load tests settle them.

## Tonight's runner: one script, short cases, prod restored at the end
- **Files:** `Storage/m6000-llama-cpp/opt27b-20261113/quick.sh`, writing to `LEDGER.md` (one row per case). Drop the long `bench.py`, or keep it only for later.
- **Container:** the docker block from `exp-27b-serving-20261106.sh` (exp container on :11448, same image, sm_52 env flags, gate-copy .so).
- **Before the first case:** stop occamy with `docker stop llamacpp-m6000` (no rm).
- **Each case:**
  - stop prod;
  - start the exp container and wait at most 3 min for health;
  - read VRAM, `/slots` n_ctx and the load log;
  - measure pp4k (random tokens), tg256 with MTP at temp 0 including draft accept, and 3 fixed temp-0 probe completions compared with case P;
  - remove the exp container (`rm -f`).

  That is about 4–5 min per case. A failed load counts as a fit result, not an error.
- **Time limit:** 90 min hard cap. A trap always runs `docker start llamacpp-m6000-27b` + health + `labslot check`.

## Cases (in order; quick gains first)
1. **P:** prod repro, the control.
2. **Fit layouts** (at the prod ub):
   - **F1:** -np 2, 2×128k, q4_0 KV
   - **F2:** -np 2, 256k `--kv-unified` q4_0
   - **F3:** -np 2, 2×96k q8
   - **F4:** F1 + mmproj, to check whether vision fits
3. **Runtime levers on the best fitting layout:**
   - **U1:** ub 1024
   - **U2:** ub 2048
   - **M1:** MTP with draft max set lower or higher (e.g. 2 and 4 against the default)
   - **K1:** draft KV q8 (`-ctkd/-ctvd`)
   - **G1:** CUDA graphs off or on
   - **E1:** each sm_52 env flag that the MoE work tuned, one case each, but only where the flag names a code path a dense model uses (MMQ_*, Q8_REUSE, NORM_Q8)
4. **Model quant:** Q4_0-head6 against IQ4_XS (15.4 GB) on the best layout. Maxwell IQ4 kernels are likely slower; this row confirms it.

## Keep or revert
- **Keep a row** only if it beats the current best on tg or pp4k by at least 3%, with no fit regression and probes identical or at least 95% token-prefix match.
- **Adopting into prod:** this happens tomorrow in daylight, by hand. Write a new run-solo-27b script, keep the old one as rollback, then `labslot set/check`.

## Whole night, cheapest execution
- **Self-driving loop:** `quick.sh --round N` runs one round of 90 min or less, then restores prod.
  - The script makes every keep-or-revert decision with the rule below, a greedy search one lever at a time.
  - Each round starts from the best row in LEDGER.md and tries the next untried levers.
  - No model judges anything.
- **Rounds:**
  - R1 is fit layouts plus P.
  - R2 is ub/MTP/graphs on the best layout.
  - R3 is env flags and draft KV.
  - R4 is the quant comparison, plus re-measuring the best row 3× for noise.
  - Between rounds, prod is up for 15 min, so the M6000 is never held for long.
- **Monitor:** one Sonnet background subagent (cheap) chains the rounds.
  - Its prompt is small: run R1..R4 in the background and wait for each to finish.
  - If a round fails (container won't start, trap fired), it checks that prod is healthy and skips to the next round.
  - It never edits prod config.
  - At the end it writes `MORNING.md` (≤20 lines: best layout, tg/pp vs baseline, vision fit, next lever) and returns a 5-line conclusion.
  - I only read that conclusion.
- **Local models:** they keep running prod on their own between rounds. The board and Hermes are not touched.

## Side jobs tonight (CPU only)
- Repoint lab-slotkeeper to :11437 / llamacpp-m6000-27b (backup .bak-20261106-27b first).
- Save the owner feedback to memory: short M6000 sessions, quick wins first, no elaborate structure.

## Verification
- `LEDGER.md` has one row per case: layout, VRAM, n_ctx, pp4k, tg, accept rate, probe match.
- Prod 27B is healthy on :11437 and `labslot check` is OK after the runner exits.
- Morning summary: best fitting 2-slot layout, tg against the 21 t/s target, vision fit yes or no, next lever for the loop.
