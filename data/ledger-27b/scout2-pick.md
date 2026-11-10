# Scout 2 pick: cut re-prefill on agent turns (checkpoint placement)
Why: real board logs show prefill at 29% and 38% of slot time (older builds, slot times overlap), mostly in evals of 2k-30k tokens. Checkpoint restores hit 1887/3522 and 1703/1851 evals. The hybrid model cannot trim the cache mid-prompt, so a mismatch rolls back to the last checkpoint and tokens already seen get prefilled again. Design 2 cut decode time, so prefill's share grows.
Unknown, and the point of stage 0: what part of those tokens is re-processed rather than new. Nothing below has been measured yet.
## Stage 0: log replay (haiku card, no GPU, ~15 min)
- Read-only `docker logs` of llamacpp-m6000 and llamacpp-m6000-RETIRED-card16-20261027.
- Per slot and task: new prompt task.n_tokens, cached/restored n_past, prior task's end n_tokens, prompt eval tokens and ms.
- Output: re-processed tokens = prior end n_tokens - restored n_past, split into generated reply vs older prompt; share of prefill minutes; prompt depth of each restore.
- Also tally evals where the mismatch came before the last reply, because a post-reply checkpoint cannot help those.
- GO if re-processed >= 25% of prefill minutes. Otherwise stop and run the runner-up (draft 3 row below).

## Stage 1: flags bench (sonnet card, GPU, ~1 h)
- Lib mmvqT-fagemm-fadec2.so, DEC=2, exp container on port 11448, hold gpu-bench.lock.
- Workload: 20-turn agent replay, about 30k start ctx. Use a real Hermes session if one is on disk, otherwise synthetic with tool tails of 2-8k tokens and 300-token replies. Temp 0, cache_prompt true.
- Arms: base `--ctx-checkpoints 8 --checkpoint-min-step 2048`; A `32 / 2048`; B `32 / 512`; C `64 / 256`. Run each twice.
- Record: total wall per session, prompt_n per turn, restores, host RSS, time cost per checkpoint (149.6 MiB each at short ctx, from the prod log).

## Stage 2: only if flags leave most of the re-processed share (sonnet card, server C++)
- In a copied source tree, add a checkpoint at the end of generation and just before the last assistant turn, behind an env flag. No kernel work.
- Re-run the Stage 1 workload as a new arm.

## Expected (ESTIMATE)
Agent session wall -10-15% at about 30k ctx, more at deep ctx. Short tg and pp unchanged. This does not reduce the first full prefill.

## Gate
- Keep if session wall is <= -10% vs base over both runs.
- tg1 and pp4k >= -1% (quick.py), host RSS growth <= 6 GiB, no restore failures.
- Stage 2 only: final-turn outputs at temp 0 match the base arm, or differences are explained.

## Runner-up if Stage 0 says NO-GO (haiku card)
Draft 3 + DEC=2 at 51.9k and 100k via tg_long_fadec2.py and DRAFT=3 (~1 h). Estimate: +10% filler @100k, about -2% real text, -10% short (measured earlier on an older base).
