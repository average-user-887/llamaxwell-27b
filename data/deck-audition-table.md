# Deck audition results (vs qwen35-9b-prod baseline)

Notes: NIAH = passes/17 probes (m4k x5, s4k x5, m16k x2, s16k x2, m32k, s32k, s60k). IFEval = inst strict/loose | prompt strict/loose (limit 100). BFCL = mean of 4 partial categories (simple_python, multiple, parallel, multi_turn_base; 20-30 cases each, so noisy). pp/tg = mean of the 2 single-slot runs (~2.8k prompt, 128 gen). Peak = gtt_mib. Marks: (+) better, (=) equal (within ~3%), (-) worse.

| model | NIAH | IFEval inst s/l | IFEval prompt s/l | BFCL mean (simple/mult/par/multi-turn) | pp t/s | tg t/s | peak MiB |
|---|---|---|---|---|---|---|---|
| qwen35-9b-prod (base) | 17/17 | .896/.920 | .850/.870 | 85.0 (95/100/80/65) | 108.1 | 12.2 | 6659 |
| lfm25-8b-a1b | 14/17 (-) fails m16k, m32k, s4k | .540/.546 (-) | .520/.530 (-) | 82.1 (95/96.7/86.7/50) (-) | 154.3 (+) | 54.1 (+) | 5340 (+) |
| mellum21-12b-a25b | 15/17 (-) fails m16k, s60k | .865/.865 (-) | .870/.870 (=/+) | 82.9 (95/90/86.7/60) (-) | 111.9 (=) | 29.4 (+) | 6870 (-, +3%) |
| qwen36-14b-a3b-fv | 17/17 (=) | .749/.755 (-) | .640/.650 (-) | 63.3 (95/96.7/46.7/15) (-) | 88.1 (-) | 22.5 (+) | 6976 (-); guard trip rc -9 |
| ornith15-9b | 17/17 (=) | .773/.816 (-) | .680/.730 (-) | 79.2 (95/93.3/83.3/45) (-) | 113.1 (=) | 12.4 (=) | 6927 (-, +4%) |
| qwen38-9b-distill | 17/17 (=) | .902/.926 (=/+) | .850/.880 (=/+) | 79.2 (95/96.7/90/35) (-) | 112.9 (=) | 12.3 (=) | 6959 (-, +4.5%) |
| granite42-8b | n/a (load failed) | n/a | n/a | n/a | 114.0* | 13.0* | 7330; guard trip (mem 1038, psi 21 at 15 s, rc -9) |

*granite speed.json exists although load.failed/guard trip; numbers look like the prod server (identical prompt sizes) or a partial run. Do not trust.

Best per metric (excluding granite):
- NIAH: tie, prod/qwen36/ornith/qwen38 17/17. IFEval: qwen38-9b-distill (inst .902/.926, prompt .850/.880), marginally over prod.
- BFCL: prod 85.0 (multi-turn 65 vs next best 60). Speed: lfm25-8b-a1b (pp 154, tg 54, also lowest mem 5340).
- Candidates beating prod on quality without losing speed: none cleanly. qwen38-9b-distill ties NIAH, edges IFEval (+0.5 to +1 pt), same speed, but loses BFCL (multi-turn 35 vs 65) and +300 MiB.
- mellum21 and lfm25 are far faster (tg 29/54 vs 12) but lose NIAH, IFEval (lfm hugely) and BFCL; mellum is the only speed-for-quality trade worth considering.
- Verdict: prod stays deck-main on this data; qwen38 only if BFCL multi-turn is not needed (n=20, rerun to confirm).
