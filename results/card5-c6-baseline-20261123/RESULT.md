# Card 5 (23 Nov 2026): C6 as quick.py base - PARTIAL

| Metric | C6 base (xmadn8-v2) | Old base (xmad27, 22 Nov rebase) |
|---|---|---|
| pp4k t/s | 91.3 | 91.3 |
| pp23k t/s (quick) | 83.96 | 84.79 |
| tg1 t/s | 26.32 | 22.66 |
| tg2agg t/s | 24.19 | 21.39 |
| accept | 154/200 | 152/204 |
| vram peak MiB | 23219 | 23217 |
| tg_long 0.0K ctx (prompt 26) tg | 22.77 | - |
| tg_long 22.8K ctx: pp / tg | 83.88 / 22.66 | - |
| tg_long 51.9K ctx: pp / tg | 75.41 / 19.34 | - |
| tg_long 100.2K ctx: pp / tg | 64.51 / 16.13 | - |
| agentic2 two-slot makespan | NOT MEASURED (see caveats) | - |

Quick.py rebase = mean of 2 reps (BASE-rebase, QUICK_LONG=1), saved to state.json base_m.
Lib: prod-libs/libggml-cuda-xmadn8-v2-20261123.so sha256 38ae66b9df7a282bdbbbf9af310fd665ecee71e90ffe3d890595bec07bbe0062
Temps (10 s samples, card5-20261123/temps.csv): max 73 C (08:42, during tg_long/agentic start), no abort at 80 C.

## Method
- Backups: state.json.bak-20261123-c6base, quick.py.bak-20261123-c6base. state.json base.so set to the lib above; 18 env vars unchanged (4 in state + 14 in quick.py ENV), N8 default on in lib; base_m cleared then refilled by the rebase; old base+base_m under notes.old_base_20261123_pre_c6.
- chain.sh under gpu-bench.lock (flock), prod llamacpp-m6000-27b docker-stopped first: quick.py 23, tg_long.py 1200 2600 4900, agentic2.py 2 (x2).
- Declared on the dashboard via labexp start/end (chain was already running, not wrapped).

## Caveats
- agentic2.py failed in both sessions with HTTP 400 at the first prefill: calibrate() used a short tag ("probe"), 18.65 tok/rep, versus 27.73 tok/rep with real seed tags, so filler_reps=4825 gave ~134K tokens > 131072 per-slot ctx. Fixed in agentic2.py (tag "seed1234567_0_s0", n=1000, target minus 1500 tool-chunk headroom; backup agentic2.py.bak-20261123-card5) -> expected filler ~3070. The fixed script was NOT re-run: card6 took the GPU lock, and a run needs roughly 45+ min per session (2 slots x 90K prefill at ~65 t/s per rep), beyond the 90 min limit. Rerun: `python3 agentic2.py 2` under the lock.
- Prod llamacpp-m6000-27b left STOPPED (card6 holds/waits on the lock); the owner of card6 must `docker start` it afterwards.
- Old-base 22 Nov numbers are from state.json notes; tg1 differences vs old base are C6 (N8) gain plus run noise.

## Files
card5-20261123/{chain.sh,chain.log,quick23.log,tglong.log,agentic2-A.log,agentic2-B.log,temps.csv}
