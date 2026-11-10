# opt27b quick ledger

| time | case | layout | VRAM MiB | pp4k | tg1 | tg2agg | accept | probe | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 11-06 23:39 | P | 1x131072 q8_0 ub512 | 21526 | 72.3 | 16.18 |  | 162/275 |  | control  |
| 11-06 23:39 | F5-2x128k-q8 | 2x131072 q8_0 ub512 |  |  |  |  |  |  | NOFIT/FAIL  |
| 11-06 23:41 | F1-2x128k-q4 | 2x131072 q4_0 ub512 | 23405 | 72.0 | 16.38 | 16.89 | 158/291 | 0.61 | FITS  |
| 11-06 23:44 | F2-256k-unified-q4 | 2x262144u q4_0 ub512 | 23661 | 71.3 | 16.35 | 16.79 | 158/291 | 0.61 | FITS  |
| 11-06 23:44 | F4-vision | 2x131072 q4_0 ub512 | 24188 |  |  |  |  | None | VISION-NOFIT  |
| 11-07 00:02 | U1-ub1024 | 2x131072 q4_0 ub1024 | 23829 | 79.1 | 16.53 | 16.85 | 158/291 | 1.0 | KEEP tg +0.9% pp +9.9% |
| 11-07 00:03 | U2-ub2048 | 2x131072 q4_0 ub2048 | 24329 |  |  |  |  | None | revert error |
| 11-07 00:05 | M2-draft2 | 2x131072 q4_0 ub1024 | 23530 | 78.5 | 17.37 | 16.75 | 141/226 | 1.0 | KEEP tg +5.1% pp -0.8% |
| 11-07 00:07 | M4-draft4 | 2x131072 q4_0 ub1024 | 24128 | 77.9 | 13.71 | 0.0 | 166/355 | None | revert error |
| 11-07 00:07 | M6-draft6 | 2x131072 q4_0 ub1024 | 24377 |  |  |  |  | None | revert error |
| 11-07 00:10 | K1-draftkv-q8 | 2x131072 q4_0 ub1024 | 24098 | 78.3 | 17.32 | 16.57 | 140/228 | 1.0 | revert tg -0.3% pp -0.3% |
| 11-07 00:12 | G0-graphs-off | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.39 | 17.54 | 141/226 | 1.0 | revert tg +0.1% pp -0.5% |
| 11-07 00:29 | E-rms_warp1 | 2x131072 q4_0 ub1024 | 23530 | 79.7 | 17.49 | 17.41 | 150/210 | 0.667 | revert probe 0.667 |
| 11-07 00:32 | E-mmq_compact0 | 2x131072 q4_0 ub1024 | 23530 | 78.3 | 17.43 | 17.51 | 141/226 | 1.0 | revert tg +0.3% pp -0.3% |
| 11-07 00:34 | E-mmq_xmad0 | 2x131072 q4_0 ub1024 | 23530 | 78.3 | 17.42 | 17.64 | 141/226 | 1.0 | revert tg +0.3% pp -0.3% |
| 11-07 00:36 | E-mmq_tail0 | 2x131072 q4_0 ub1024 | 23530 | 78.2 | 17.41 | 17.49 | 141/226 | 1.0 | revert tg +0.2% pp -0.4% |
| 11-07 00:39 | E-q8reuse0 | 2x131072 q4_0 ub1024 | 23528 | 78.1 | 17.34 | 17.62 | 141/226 | 1.0 | revert tg -0.2% pp -0.5% |
| 11-07 00:41 | E-normq8_0 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.4 | 17.47 | 141/226 | 1.0 | revert tg +0.2% pp -0.5% |
| 11-07 00:43 | E-rowpack0 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.39 | 17.64 | 141/226 | 1.0 | revert tg +0.1% pp -0.5% |
| 11-07 00:46 | E-occfa0 | 2x131072 q4_0 ub1024 | 23530 | 78.2 | 17.41 | 17.53 | 141/226 | 1.0 | revert tg +0.2% pp -0.4% |
| 11-07 00:48 | E-occfused0 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.39 | 17.57 | 141/226 | 1.0 | revert tg +0.1% pp -0.5% |
| 11-07 00:50 | E-force_mmq | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.41 | 17.51 | 141/226 | 1.0 | revert tg +0.2% pp -0.5% |
| 11-07 00:53 | E-force_cublas | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.4 | 17.55 | 141/226 | 1.0 | revert tg +0.2% pp -0.5% |
| 11-07 01:11 | Q-IQ4_XS | 2x131072 q4_0 ub1024 | 22296 | 78.5 | 14.41 | 15.46 | 153/203 | 0.402 | QUANT-CANDIDATE probe 0.402 |
| 11-07 01:14 | Q-Q3_K_XL | 2x131072 q4_0 ub1024 | 21241 | 76.1 | 13.82 | 15.46 | 156/196 | 0.531 | QUANT-CANDIDATE probe 0.531 |
| 11-07 01:16 | B-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.5 | 17.42 | 17.57 | 141/226 | 1.0 | noise  |
| 11-07 01:19 | B-rep1 | 2x131072 q4_0 ub1024 | 23530 | 78.3 | 17.4 | 17.56 | 141/226 | 1.0 | noise  |
| 11-07 01:21 | B-rep2 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.39 | 17.59 | 141/226 | 1.0 | noise  |
| 11-07 14:26 | G1-maxwell-graphs | 2x131072 q4_0 ub1024 | 23544 | 79.7 | 17.56 | 17.67 | 141/226 | 1.0 | revert tg +1.1% pp +1.5% |
| 11-07 14:28 | G2-graph-opt | 2x131072 q4_0 ub1024 | 23550 | 78.3 | 17.4 | 16.67 | 141/226 | 1.0 | revert tg +0.2% pp -0.3% |
| 11-07 14:30 | X1-maxwell-fusion | 2x131072 q4_0 ub1024 | 23530 | 78.2 | 17.36 | 17.59 | 141/226 | 1.0 | revert tg -0.1% pp -0.4% |
| 11-07 14:33 | X2-mmvq-pipe | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.43 | 17.63 | 140/228 | 1.0 | revert tg +0.3% pp -0.5% |
| 11-07 14:35 | A1-native-fa-sm52 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.37 | 17.55 | 141/226 | 1.0 | revert tg +0.0% pp -0.5% |
| 11-07 14:37 | A2-fa-vec | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.48 | 17.63 | 140/228 | 1.0 | revert tg +0.6% pp -0.5% |
| 11-07 14:40 | A3-fa-waves2 | 2x131072 q4_0 ub1024 | 23530 | 78.2 | 17.36 | 17.55 | 141/226 | 1.0 | revert tg -0.1% pp -0.4% |
| 11-07 14:42 | A4-fa-waves4 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.39 | 17.56 | 141/226 | 1.0 | revert tg +0.1% pp -0.5% |
| 11-07 14:44 | D1-gdn-dec1 | 2x131072 q4_0 ub1024 | 23530 | 78.2 | 17.36 | 17.57 | 142/224 | 1.0 | revert tg -0.1% pp -0.4% |
| 11-07 14:47 | D4-gdn-dec4 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.37 | 17.61 | 140/228 | 1.0 | revert tg +0.0% pp -0.6% |
| 11-07 14:49 | X0-fusion-off | 2x131072 q4_0 ub1024 | 23530 | 77.6 | 16.97 | 17.12 | 141/226 | 1.0 | revert tg -2.3% pp -1.1% |
| 11-07 14:51 | M1-draft1 | 2x131072 q4_0 ub1024 | 23230 | 78.7 | 17.34 | 18.45 | 115/140 | 0.491 | revert probe 0.491 |
| 11-07 14:54 | M3-draft3 | 2x131072 q4_0 ub1024 | 23829 | 77.8 | 16.33 | 16.84 | 158/291 | 1.0 | revert tg -6.0% pp -0.9% |
| 11-07 14:56 | S1-pmin0.6 | 2x131072 q4_0 ub1024 | 23530 | 77.8 | 17.2 | 17.68 | 132/163 | 0.764 | revert probe 0.764 |
| 11-07 14:59 | S2-pmin0.3 | 2x131072 q4_0 ub1024 | 23530 | 77.8 | 17.3 | 17.31 | 153/197 | 0.829 | revert probe 0.829 |
| 11-07 15:01 | U3-ub512 | 2x131072 q4_0 ub512 | 23106 | 71.5 | 17.4 | 17.56 | 141/226 | 1.0 | revert tg +0.2% pp -8.9% |
| 11-07 15:03 | U4-ub768 | 2x131072 q4_0 ub768 | 23318 | 75.2 | 17.34 | 16.68 | 141/226 | 1.0 | revert tg -0.2% pp -4.2% |
| 11-07 15:06 | U5-ub1536 | 2x131072 q4_0 ub1536 | 23956 | 77.5 | 17.4 | 17.53 | 141/226 | 1.0 | revert tg +0.2% pp -1.3% |
| 11-07 15:08 | B1-b1024 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.41 | 17.65 | 141/226 | 1.0 | revert tg +0.2% pp -0.5% |
| 11-07 15:08 | K2-k8v4 | 2x131072 q4_0 ub1024 |  |  |  |  |  |  | NOFIT/FAIL  |
| 11-07 15:10 | B6-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.3 | 17.38 | 17.53 | 141/226 | 1.0 | noise  |
| 11-07 15:13 | B6-rep1 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.32 | 16.69 | 141/226 | 1.0 | noise  |
| 11-07 15:15 | G1b-maxwell-graphs | 2x131072 q4_0 ub1024 | 23544 | 78.8 | 17.42 | 17.62 | 141/226 | 1.0 | revert tg +0.3% pp +0.4% |
| 11-07 15:18 | A2b-fa-vec | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.46 | 17.53 | 140/228 | 1.0 | revert tg +0.5% pp -0.6% |
| 11-07 15:20 | X2b-mmvq-pipe | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.41 | 17.56 | 140/228 | 1.0 | revert tg +0.2% pp -0.5% |
| 11-07 15:22 | X1b-maxwell-fusion | 2x131072 q4_0 ub1024 | 23530 | 78.2 | 17.31 | 17.5 | 141/226 | 1.0 | revert tg -0.3% pp -0.4% |
| 11-07 15:25 | B8-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.36 | 17.56 | 141/226 | 1.0 | noise  |
| 11-07 15:27 | B8-rep1 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.36 | 17.49 | 141/226 | 1.0 | noise  |
| 11-07 15:29 | B8-rep2 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.38 | 17.53 | 141/226 | 1.0 | noise  |
| 11-07 15:30 | V1-vision-base | 2x131072 q4_0 ub1024 | 24312 |  |  |  |  | None | NOFIT  |
| 11-07 15:32 | V2-vision-ub512 | 2x131072 q4_0 ub512 | 24238 | 71.7 | 17.36 | 17.49 | 141/226 | 1.0 | NOFIT  |
| 11-07 15:35 | V3-vision-2x96k-q4 | 2x98304 q4_0 ub1024 | 22870 | 78.1 | 17.32 | 17.54 | 141/226 | 1.0 | FITS  |
| 11-07 15:35 | L1-2x96k-q8 | 2x98304 q8_0 ub1024 | 24460 |  |  |  |  | None | NOFIT  |
| 11-07 15:37 | C1-ckpt0 | 2x131072 q4_0 ub1024 | 23530 | 79.3 | 18.24 | 17.89 | 149/210 | 0.353 | revert probe 0.353 |
| 11-07 15:40 | C2-cacheram0 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.36 | 17.62 | 141/226 | 1.0 | revert tg -0.1% pp -0.6% |
| 11-07 15:42 | T1-threads4 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.34 | 17.5 | 141/226 | 1.0 | revert tg -0.2% pp -0.5% |
| 11-07 15:44 | T2-prio2 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.37 | 17.46 | 141/226 | 1.0 | revert tg +0.0% pp -0.5% |
| 11-07 15:47 | B7-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.35 | 17.49 | 141/226 | 1.0 | noise  |
| 11-07 15:49 | B7-rep1 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.36 | 17.55 | 141/226 | 1.0 | noise  |
| 11-07 15:51 | B7-rep2 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.35 | 17.55 | 141/226 | 1.0 | noise  |
| 11-07 15:54 | K-mmvq-base | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.36 | 17.52 | 141/226 | 1.0 | revert tg -0.1% pp -0.5% |
| 11-07 15:56 | K-mmvq-w2 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.33 | 17.56 | 140/228 | 1.0 | revert tg -0.2% pp -0.6% |
| 11-07 15:58 | K-mmvq-w4 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.99 | 16.73 | 151/206 | 0.785 | revert probe 0.785 |
| 11-07 16:01 | K-mmvq-r1 | 2x131072 q4_0 ub1024 | 23530 | 77.8 | 13.45 | 12.81 | 141/226 | 1.0 | revert tg -22.6% pp -0.9% |
| 11-07 16:04 | K-mmvq-w4r1 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 14.09 | 13.35 | 151/206 | 0.785 | revert probe 0.785 |
| 11-07 16:06 | K-mmvq-up | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.82 | 17.07 | 151/206 | 0.785 | revert probe 0.785 |
| 11-07 16:08 | B9-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.36 | 17.54 | 141/226 | 1.0 | noise  |
| 11-07 16:11 | B9-rep1 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.36 | 17.55 | 141/226 | 1.0 | noise  |
| 11-07 16:19 | Q-head4 | 2x131072 q4_0 ub1024 | 23217 | 78.6 | 19.81 | 17.8 | 153/203 | 0.359 | CANDIDATE step_ms 125.5 vs 128.2 (-2.1%) |
| 11-07 16:21 | K-mmvq-n3w4 | 2x131072 q4_0 ub1024 | 23530 | 78.3 | 17.25 | 17.31 | 141/226 | 0.491 | revert step_ms 129.0 vs 128.2 (+0.6%) |
| 11-07 16:24 | K-mmvq-n58w4 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.41 | 17.49 | 141/226 | 1.0 | revert step_ms 127.9 vs 128.2 (-0.2%) |
| 11-07 16:26 | K-mmvq-n24w4 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 17.96 | 16.25 | 151/206 | 0.785 | revert step_ms 135.8 vs 128.2 (+5.9%) |
| 11-07 16:28 | B10-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 17.41 | 17.5 | 141/226 | 1.0 | noise step_ms 127.9 |
| 11-07 16:32 | K-mmvq-p4n1 | 2x131072 q4_0 ub1024 | 23530 | 78.5 | 19.67 | 19.29 | 141/226 | 1.0 | KEEP tg +13.2% pp +0.0% |
| 11-07 16:34 | K-mmvq-p4 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 19.38 | 19.34 | 141/226 | 1.0 | revert tg -1.5% pp -0.6% |
| 11-07 16:36 | K-mmvq-p4n4 | 2x131072 q4_0 ub1024 | 23530 | 78.1 | 19.73 | 19.33 | 141/226 | 1.0 | revert tg +0.3% pp -0.5% |
| 11-07 16:38 | B11-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.0 | 19.69 | 19.35 | 141/226 | 1.0 | noise step_ms 113.1 |
| 11-07 16:42 | M3b-draft3-p4n1 | 2x131072 q4_0 ub1024 | 23829 | 78.1 | 17.6 | 18.8 | 158/291 | 1.0 | revert tg -10.5% pp -0.5% |
| 11-07 16:43 | M4b-draft4-p4n1 | 2x131072 q4_0 ub1024 | 24128 | 77.9 | 15.7 | 0.0 | 166/355 | None | revert error |
| 11-07 16:46 | Q-head4-p4n1 | 2x131072 q4_0 ub1024 | 23217 | 78.3 | 22.44 | 19.91 | 153/203 | 0.359 | CANDIDATE step_ms 110.8 vs 113.2 (-2.1%) |
| 11-07 16:48 | B12-rep0 | 2x131072 q4_0 ub1024 | 23530 | 78.3 | 19.66 | 19.35 | 141/226 | 1.0 | noise step_ms 113.2 |
| 11-07 16:59 | Q-head4-adopt | 2x131072 q4_0 ub1024 | 23217 | 79.1 | 22.49 | 20.03 | 153/203 | 0.359 | KEEP KLD 0.0022 same-top 98.6%; step_ms 110.5 vs 113.2 |
| 11-07 17:01 | B13-rep0 | 2x131072 q4_0 ub1024 | 23217 | 78.8 | 22.48 | 19.99 | 153/203 | 1.0 | noise step_ms 110.6 |
| 11-07 17:03 | B13-rep1 | 2x131072 q4_0 ub1024 | 23217 | 78.1 | 22.43 | 19.94 | 153/203 | 1.0 | noise step_ms 110.8 |
| 11-07 17:06 | M1b-draft1-head4 | 2x131072 q4_0 ub1024 | 22918 | 78.5 | 20.76 | 20.56 | 114/141 | 0.521 | revert tg1 20.76 vs 22.49; tg2agg 20.56 vs 20.03; step_ms 86.8 |
| 11-07 17:08 | B14-rep0 | 2x131072 q4_0 ub1024 | 23217 | 78.2 | 22.45 | 19.93 | 153/203 | 1.0 | noise step_ms 110.7 |
| 11-07 17:48 | K-fp32mv | 2x131072 q4_0 ub1024 | 23217 | 79.7 | 20.95 | 21.26 | 145/218 | 0.476 | revert tg1 20.95 vs 22.49; tg2agg 21.26 vs 20.03; step_ms 110.1 vs 110.5 |
| 11-07 17:50 | K-fp32mv-draft3 | 2x131072 q4_0 ub1024 | 23516 | 78.3 | 20.49 | 18.95 | 159/285 | 0.476 | revert tg1 20.49 vs 22.49; tg2agg 18.95 vs 20.03; step_ms 128.8 vs 110.5 |
| 11-07 17:52 | B15-rep0 | 2x131072 q4_0 ub1024 | 23217 | 78.3 | 22.45 | 19.95 | 153/203 | 1.0 | noise step_ms 110.7 |
| 11-07 18:07 | P-sgemmnn | 2x131072 q4_0 ub1024 | 23217 | 87.0 | 22.45 | 21.01 | 152/204 | 0.833 | revert probe 0.833 |
| 11-07 18:09 | B16-rep0 | 2x131072 q4_0 ub1024 | 23217 | 79.0 | 22.48 | 19.99 | 153/203 | 1.0 | noise pp4k 79.0 step_ms 110.6 |
| 11-07 18:17 | P-sgemmnn-adopt | 2x131072 q4_0 ub1024 | 23217 | 87.2 | 22.39 | 21.04 | 152/204 | 0.833 | KEEP KLD ~0 same-top 100%; pp4k 87.2 vs 79.1; tg1 22.39 vs 22.49 |
| 11-07 18:19 | B17-rep0 | 2x131072 q4_0 ub1024 | 23217 | 86.2 | 22.34 | 20.95 | 152/204 | 1.0 | noise pp4k 86.2 step_ms 110.2 |
| 11-07 18:28 | P-dqT2 | 2x131072 q4_0 ub1024 | 23217 | 89.2 | 22.35 | 21.48 | 152/204 | 1.0 | KEEP tg -0.2% pp +2.3% |
| 11-07 18:30 | B18-rep0 | 2x131072 q4_0 ub1024 | 23217 | 89.0 | 22.32 | 21.43 | 152/204 | 1.0 | noise pp4k 89.0 step_ms 110.3 |
| 11-07 18:36 | P-nnsplit4 | 2x131072 q4_0 ub1024 | 23217 | 87.2 | 22.35 | 21.49 | 152/204 | 1.0 | revert tg +0.0% pp -2.2% |
| 11-07 18:38 | P-nnsplit2 | 2x131072 q4_0 ub1024 | 23217 | 87.8 | 22.32 | 21.46 | 152/204 | 1.0 | revert tg -0.1% pp -1.6% |
| 11-07 18:40 | B19-rep0 | 2x131072 q4_0 ub1024 | 23217 | 88.8 | 22.26 | 21.45 | 152/204 | 1.0 | noise pp4k 88.8 step_ms 110.6 |
| 11-07 19:21 | C1b-ckpt0-now | 2x131072 q4_0 ub1024 | 23217 | 91.7 | 22.45 | 21.79 | 157/196 | 0.618 | TRADEOFF tg1 22.45 vs 22.35; tg2agg 21.79 vs 21.48; pp4k 91.7 vs 89.2 |
| 11-07 20:20 | FA-gemm (GGML_CUDA_FA_GEMM=1, lib mmvqT-fagemm) | F1-2x128k-q4 | | 86.6 (vs 86.9 @2.2k) | | | | KLD 0 / max 5.3e-5 / top 100%; tbo 3993/3993 | KEEP-candidate: pp@22.7k 83.73 vs 77.98 (+7.4%) |
| 11-07 21:37 | P-fagemm-adopt | 2x131072 q4_0 ub1024 | 23217 | 90.8 | 22.32 | 21.41 | 152/204 | 1.0 | KEEP long-ctx pp +7/+15/+24% @22.7k/51.9k/100.2k; pp4k 90.8 vs 89.2; tg1 22.32 vs 22.35 |
| 11-07 22:50 | CLK1152-rep0 | 2x131072 q4_0 ub1024 | 23217 | 91.2 | 22.26 | 21.46 | 152/204 | 1.0 | revert tg -0.3% pp +0.4% vs 987MHz pp4k 90.8 tg1 22.32 tg2agg 21.41 |
| 11-07 22:52 | CLK1152-rep1 | 2x131072 q4_0 ub1024 | 23217 | 91.2 | 22.27 | 21.46 | 152/204 | 1.0 | revert tg -0.2% pp +0.4% vs 987MHz pp4k 90.8 tg1 22.32 tg2agg 21.41 |

## Scout (Fable, 2026-11-07) - ANALYTIC / UNMEASURED, no GPU used
| time | case | layout | VRAM MiB | pp4k | tg1 | tg2agg | accept | probe | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 11-07 23:00 | S1-tg-roofline (analytic) | C3 | | | 22.3 | | 152/204 | | Q4_0 27B ~15.2 GB/step @ ~250 GB/s eff = ~61 ms floor (+~6 ms 2x MTP/lm_head) vs step_ms 110.5 measured -> decode at ~55-60% of BW roofline; ~40 ms/step unexplained = biggest lever |
| 11-07 23:00 | S2-draft-slope (analytic, ledger) | C3 | | | | | | | draft1 step 86.8 -> draft2 110.5: +23.7 ms per extra verify column; a BW-bound matvec should add ~0. Extra column costs ~27 GMAC -> decode verify is ALU-bound, not memory-bound |
| 11-07 23:00 | S3-sm52-dp4a (source grep) | src-b11552 common.cuh:739-745 | | | | | | | sm_52 < DP4A cc: ggml_cuda_dp4a falls back to 4 byte-extract+imul+add per int32 -> ~3-4 ALU ops/weight/column; 3 cols x 27G x ~3.5 ops / ~3.2 T ops/s (~1.04 GHz loaded) = ~80 ms, matches the gap |
| 11-07 23:00 | S4-fp32mv-recheck (ledger) | round15 | | | 20.95 | 21.26 | 145/218 | | op 214->194 us (n=3) but step 110.5->110.1: kernel win hidden; tg1 loss came from accept 0.75->0.67 (numerics shift), not speed. Lesson: new decode kernel must keep q8_1 numerics bit-exact or be accept-gated |
| 11-07 23:00 | S5-longctx-tg-KV (analytic, assumed 16 full-attn layers, 4 KV heads, d256) | 2x128k q4_0 KV | | | | | | | ~18 KB/token KV -> 1.8 GB/step at 100k = +7 ms (~-6% tg) if decode FA is BW-efficient; larger drop in the in-flight tg@100k row means decode-FA kernel, not KV bytes |
| 11-07 23:00 | S6-clock (ledger r22) | | | | | | | | power-capped at 250 W, ~1037 MHz loaded: ALU-bound decode cannot buy clocks -> must cut ops/weight (energy) instead; supports S3 lever |
| 11-07 23:00 | RECOMMEND overnight | C3 | | | | | | | sm_52 decode matvec: cut ALU ops/weight in q4_0 x q8_1 mmvq for ncols 1-3 (prmt/__byte_perm nibble unpack + __vmul-free XMAD pairs, or 16-bit packed int sums via __vadd/__vsadu-free paths; dequant once per weight shared across columns; keep integer q8_1 result bit-exact). Expected +10-20% tg1/tg2agg (step 110 -> ~90-100 ms), risk medium (fp32mv showed op gains can hide; must profile per-step first). Card: (1) nvprof decode-only, short ctx, C3, 64 tokens: share of mul_mat_vec_q n=3 in step; stop if <40%. (2) microbench kernel variant vs base with test-backend-ops -o MUL_MAT perf (n=1,2,3, q4_0, 5120x17408 + 17408x5120); keep if op time -15%. (3) bit-exact check (test-backend-ops + KLD == 0), (4) quick.py round with accept unchanged. #2 adaptive draft len (cheap, 0-3%). #3 decode FA split-K only if in-flight tg@100k drops >10% |
| 11-07 23:46 | BASE-rebase | 2x131072 q4_0 ub1024 | 23217 | 91.3 | 22.72 | 21.4 | 152/204 | 1.0 | rebase old pp4k 90.8 tg1 22.32 |
| 11-07 23:59 | BASE-rebase | 2x131072 q4_0 ub1024 | 23217 | 91.3 | 22.66 | 21.39 | 152/204 | 1.0 | rebase old pp4k 91.3 tg1 22.72 |

## 8 Nov overnight (orchestrator)
- clock sweep (chain1): 987 vs 1152 same session, pp23k +0.96%, others within noise, peak 75 °C. KEEP 1152 (apps clocks 3305,1152).
- tg long ctx @1152 C3: 18.9 / 11.58 / 8.07 t/s at 26 / 51.9k / 100k tokens. Decode collapses with ctx.
- nvprof C3 22.7k (prof-c3-50k-summary.txt): prefill sgemm 79.3%, FA 9.6%, dequant 5.6%; decode mmvq 58.8%, flash_attn_tile 22.5%, dequantize_block_q4_0<half> 11.3% (KV converted to f16 for the tile kernel?), GDN 2.8%.
- FA-gemm S16 (f16 S scratch, libggml-cuda-mmvqT-fagemm-s16.so): tbo 3993/3993 both, KLD mean 1e-6 top 99.951%, pp22.7k -0.2%, pp51.9k +0.5% (within noise). REJECT, S traffic is not the bottleneck.
| 11-08 00:40 | fadec diagnosis | C3 | | | | | | | sm_52 generic path: Q rows 3 (MTP verify) -> TILE (needs f16 K/V: whole q4_0 cache dequantized per step; prof: tile 22.5% + dequant 11.3% @22.7k); Q rows <=2 -> VEC (cols 1/2 only). D=256, GQA grouped per kv-head | |
| 11-08 00:41 | fadec design 1 (GGML_CUDA_FA_DEC=1, lib mmvqT-fagemm-fadec) | C3 | | | | | | | route Q rows 3-4 quantized KV to vec kernel, new cols_per_block=4 (fattn.cu/fattn-common.cuh/fattn-vec.cuh, .bak-20261108-fadec) | |
| 11-08 | fadec tbo | | | | | | | | FLASH_ATTN_EXT 3993/3993 for DEC=0, DEC=1, DEC=1+GEMM_MINQ1 | |
| 11-08 | fadec KLD | | | | | | | | DEC=1 c512: mean 0 / max 5.3e-5 / top 100%; DEC=1 -b3 -ub3 (exercises path): mean 0.00706 / max 6.56 / top 98.1% (no flag-0 -b3 control run) | FAIL gate |
| 11-08 | fadec tg flag0 vs 1 | | | | | | | | tg@26: 19.0 vs 18.6; @52k: 11.6 vs 8.14; @100k: 8.07 vs 5.05; pp unchanged; accept same | REJECT: vec 30-37% SLOWER (24 q-heads each stream KV, no GQA sharing; slower than tile+dequant). See fadec-ab-20261108.txt |
| 11-08 | sm52 mmvq design1 build | | | | | | | | int16-pair XMAD path for q4_0 x q8_1 ncols 2-4, GGML_CUDA_MMVQ_SM52; attempt 1 (PRMT sign-extend) FAILED tbo n=2..4 (ERR ~1.0); attempt 2 (plain sign-extend) fixed it. No prof-c3 summary: gate = tbo perf only | |
| 11-08 | sm52 tbo MUL_MAT | | | | | | | | SM52=1: 0 FAIL, 1103 OK, 61 q4_0 OK (23 at n=2..4); run aborts at q4_K n=9 GGML_ASSERT(convert_func) (unrelated; flag-0 control only checked for FAIL lines) | PASS |
| 11-08 | sm52 tbo perf q4_0 4096x14336 | | | | | | | | us flag0->flag1 (2 reps): n=1 146->144 (2%, untouched path, noise), n=2 166.6->160.9 (-3.4%), n=3 218.0->211.0 (-3.2%) | FAIL gate (<15%): STOP. Bit-exact decode, KLD, sm52_ab not run (scripts kld_sm52.sh, sm52_ab.py ready) |
| 11-08 | fadec design 2 spec | | | | | | | | GQA-grouped q4_0 decode FA (FA_DEC=2), spec only, not built (time) | see fadec-design2-spec.md |
| 11-08 | card E draft=2 (base) | C3 | | | | | | | tg 19.13 @26 / 11.59 @51.9k / 8.07 @100k; accept 69/115, 79/95, 80/94 | tg_long_draft.py DRAFT=2; out draft-long-20261108.txt |
| 11-08 | card E draft=1 | C3 | | | | | | | tg 19.14 @26 / 8.09 @51.9k / 5.20 @100k; accept 53/73, 58/68, 59/67 | REJECT vs draft=2 at long ctx (-30% @52k, -36% @100k) |
| 11-08 | card E draft=0 (MTP off) | C3 | | | | | | | tg 14.19 @26 / 5.80 @51.9k / 3.73 @100k; accept n/a | MTP off slowest at all ctx; draft=2 +35% @26, +100% @52k, +116% @100k |

## 8 Nov Card F (design 2)
Lib: mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so (build 3 of 4). Flag GGML_CUDA_FA_DEC=2; 0 and 1 unchanged. tbo, verify and MINQ=1 rows ran on build 3; KLD draws and DEC=0/2 tg rows ran on build 2. Build 3 only adds the env-gated CPU verify; the kernel default path did not change.

Gate 1 tbo FLASH_ATTN_EXT: PASS. 3993/3993 for DEC=0, DEC=2 MINQ3, DEC=2 MINQ1. With FADEC2_CASES (G=4/6/8 hsk256 q4_0): 4281/4281, fa_dec2 total calls=120 (MINQ3), 264 (MINQ1). The stock 3993 cases never reach the DEC=2 path; only the added cases do.

Gate 2 KLD -c 512 --chunks 16 -b 3 -ub 3 (pass: |d mean| <= 0.0005, same-top within 0.3 pt). 4080 scored tokens.
| draw | mean KLD | max KLD | same top | d mean vs control | gate |
|---|---|---|---|---|---|
| DEC=0 control | 0.009094 | 1.498 | 97.745 | - | - |
| DEC=2 default (shipped; 4x incl. both verify runs) | 0.010614 | 6.679 | 97.672 | +0.00152 | FAIL |
| DEC=2 SPLIT=3 | 0.010577 | 6.831 | 97.770 | +0.00148 | FAIL |
| DEC=2 SPLIT=2 | 0.009333 | 0.933 | 97.672 | +0.00024 | pass |
| DEC=2 SPLIT=1 | 0.009073 | 0.593 | 97.672 | -0.00002 | pass |
| GEMM_MINQ=1 (not DEC=2) | 0.009538 | 0.857 | 97.917 | +0.00044 | n/a |
| DEC=1 | 0.009345 | 1.197 | 97.598 | +0.00025 | n/a |
| OCCAMY_FA=1 | 0.009094 | 1.498 | 97.745 | 0 (same as control, not an independent draw) | n/a |
Standard c512 KLD: DEC=0 and DEC=2 identical (0.004511, max 1.740, top 98.309); the DEC=2 path is not exercised there.
The miss is one token of KLD ~6.7 (~0.0015 of the mean). It appears in 2 of 4 distinct DEC=2 settings and 0 of 3 distinct other paths. It is deterministic, not a race. The token was not identified.
Self-consistency on real inputs, default settings, same 6.679 token reproduced:
- VERIFY=1 (nsplit=1 re-run): ~43k calls, worst_rel 2.09e-6, 0 non-finite, 0 flagged.
- VERIFY=2 (CPU double reference, per seq/row/head): ~43k calls, 0 flagged at rel>1e-3, worst_rel 8.33e-6 and worst_abs 7.02e-5 at the last 40k progress line.
This points away from a computation error in the kernel, on calls that went through it. It does not compare with the stock kernel on the same inputs, the reference shares my reading of the inputs, and calls that fall back to stock are not checked. Rounding amplified through q4_0 KV re-quantization fits this but is UNVERIFIED. SPLIT=1/2 passing is not presented as a fix: choosing them would be selecting on noise.

Gate 3 tg / pp (2 runs each unless noted; baseline 18.9/11.58/8.07 t/s):
| ctx | tg DEC=0 | tg DEC=2 | d tg | pp DEC=0 | pp DEC=2 | d pp | accept 0 -> 2 |
|---|---|---|---|---|---|---|---|
| 26 (pooled 6+6) | 18.985 | 18.91 | -0.39% | 13.928 | 13.907 | -0.15% | 69/115 same |
| 51.9k | 11.58/11.60 | 16.78/16.77 | +44.8% | 76.44/76.30 | 76.27/76.29 | -0.12% | 79/95 -> 77/99 |
| 100k | 8.06/8.07 | 14.40/14.38 | +78% | 65.28/65.25 | 65.26/65.24 | -0.02% | 80/94 -> 79/95 |
DEC=2 MINQ=1, 1 run, not a pair: 26 ctx tg 19.07 pp 13.93 (69/115); 100k tg 14.45 pp 65.4 (78/97). MTP-off not run. maxT 75 C on the DEC=0/2 long A/B. Outputs are not bit-identical at long ctx (accept differs).
Speed criteria all pass; no SPLIT tuning was needed. The first 2+2 short runs alone were -0.61% (just over the limit); balanced extra runs give the pooled -0.39%.

Profile: nvprof unusable (error 4211:27). Knockouts at n_kv=100352 G6 NR3 on the first version: full 4.9 ms/call, no phase A 1.0, no phase C 3.56, neither 0.11, so phase A dominated (~3.6 ms, ~1/6 of peak FMA, 222 regs). Low-register phase A (AV3, 204 regs, MINB=2): 2.94 ms, maxerr OK. AV3 MINB=3: 2.85-3.70 ms, unstable, spills.

VERDICT: REJECTED as written on gate 2 (default +0.00152 > 0.0005); speed-qualified (+78% tg@100k, +44.8% @52k, -0.39% short, pp >= -0.15%). Not a clean CANDIDATE. Flagged for coordinator decision: accept the one-token miss, or first identify the token and compare it with stock. Prod not restored; NO_RESTORE left in place.

## Scout 2 (Fable, 2026-11-08)
Short scout, one GPU step (n sweep), one log read. Estimates are marked; everything else is measured.
| time | case | layout | VRAM MiB | pp4k | tg1 | tg2agg | accept | probe | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 11-08 | S2-mmvq-nsweep (tbo perf q4_0 4096x14336, SM52=0, tmp-combo/sm52/bin; single run; 53 C after) | n/a | | | | | | | n=1 144.2, n=2 162.5, n=3 214.2, n=4 274.6, n=5 336.6, n=8 491.1 us (n=6,7 not in tbo output). Slope ~60 us/col from n=2 up: no n=3 anomaly, nothing structural to fix. CLOSES mmvq padding/re-split idea; n=3 is simply past the kink. Out: scout2-mmvq-nsweep-20261108.txt |
| 11-08 | S2-step-fit (derived from fadec2 rows, steps ~= 128 - accepted) | DEC=2 | | | | | | | step 114.7 ms @26, 149.6 @51.9k, 181.5 @100k: linear, 0.67 ms per 1k KV tokens at both points |
| 11-08 | S2-decode-shares (ESTIMATE, nvprof per-kernel ms scaled to the step fit; not profiled, nvprof unusable on DEC=2) | DEC=2 @100k | | | | | | | FA ~37%, mmvq ~41-52%, rest ~11-22%, KV dequant ~0. At 26 ctx mmvq ~65-80% |
| 11-08 | S2-real-wall (docker logs, read only; llamacpp-m6000: ub512 2x96k, older build, real board work) | old prod | | | | | | | prefill 311.9 min 5.26M tok, gen 769.9 min 2.36M tok -> prefill 29% of slot time; 1887 checkpoint restores, 122 full reprocess over 3522 evals; prefill buckets <500 32.5 min, <2k 65.5, <8k 165.3, <30k 46.0, >=30k 2.6. Slot times overlap across 2 slots |
| 11-08 | S2-real-wall (same, llamacpp-m6000-RETIRED-card16-20261027) | old prod | | | | | | | prefill 425.4 min 6.42M tok, gen 707.2 min 1.03M tok -> prefill 38%; 1703 restores, 45 full reprocess over 1851 evals; buckets <500 23.0 min, <2k 50.8, <8k 132.0, <30k 153.7, >=30k 65.9 |
| 11-08 | S2-prefill-FA-roofline (ESTIMATE, analytic) | C3 @100k | | | | | | | ~0.52 s per FA call per ub1024 vs ~0.39 s at peak FMA -> already ~75% of peak; a native prefill FA kernel buys <= ~8-10% pp@100k for heavy work |
RECOMMEND: cut re-prefill on agent turns (checkpoint placement, no kernel work). First replay the real logs to size the re-processed share (no GPU), then bench flags on DEC=2. Estimated -10-15% agent turn wall, not measured. Runner-up draft 3 at long ctx. Spec: scout2-pick.md

## Scout 2 Stage 0
| time | case | layout | VRAM MiB | pp4k | tg1 | tg2agg | accept | probe | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 11-08 | S2-stage0-replay (docker logs read only; parser stage0_replay.py; results stage0-replay-20261108.txt) | old prod | | | | | | | re-processed share of prefill minutes: pooled 27.2% (A) / 27.4% (B) = GO; llamacpp-m6000 live 16.4% / 17.0% = NO-GO; RETIRED 35.4% / 35.3%. Reply vs older split pooled 32/68. Unattributable pooled: 4 no-prior + 326 (A) / 384 (B) unrelated. |

## Scout 2 Stage 1
| time | case | layout | VRAM MiB | pp4k | tg1 | tg2agg | accept | probe | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 11-08 | S2-stage1-ckpt-flags (synthetic 16k+8 turns, tails 2-4k, 40% reply / 60% earlier mismatch; seed 1 only; ckpt-flags-ab-20261108.txt) | C3 fadec2 DEC=2 | | not run | not run | | | | session wall base 736.6 s / A(32,2048) 736.9 / B(32,512) 736.9 / C(64,256) 736.8 = 0.0% for all; prompt_n per turn identical, re-processed 10410 tok = 2.27 min = 22.1% of prefill minutes in every arm (base valid: 8 restores, 18 ckpts created, 308 MiB each); RSS growth 3.8 / 4.2 / 4.6 / 4.5 GB; 0 real restore failures. GATE FAIL (wall -10% not met). Seed 2 and quick rows not run (GPU lock taken by f2/chain1.sh). |

## 8 Nov Card F2 (design 2 KLD token)
Report: fadec2-kld-token-20261108.txt. Evidence: /media/avg-usr/Models/kld-f2-20261108. Debug libs only; no fix built; fadec2.so unchanged; gate not relaxed.
| time | case | mean KLD | max | >1 | >3 | same top | result |
|---|---|---|---|---|---|---|---|
| 11-08 | DEC=0, 64 chunks b3 (16320 tokens, new base head4-64.kld, first 16 chunks byte-identical to head4.kld) | 0.011873 | 1.498 | 5 | 0 | 97.794 | |
| 11-08 | DEC=2, same | 0.012500 | 6.679 | 6 | 1 | 97.641 | paired diff +0.000627, se 0.000568, t 1.11; without 10:416 +0.000222, t 0.56 |
| 11-08 | stock + 1e-7 output noise, 16 chunks | 0.010392 | 4.500 | | 1 | | stock kernel reaches outlier at 10:416 (4.50) |
| 11-08 | DEC=2 + 1e-6 noise, 3 seeds, 16 chunks | 0.009531, 0.008518, 0.008967 | 1.696, 0.319, 1.511 | | 0 | | outlier gone at 10:416: 1.177, 0.052, 0.937 |
Token: chunk 10 pos 416 ".catch"; base .397; DEC=0 .294 (KLD .064); DEC=2 .99995 (KLD 6.679). Finite; not near-zero base; no top flip.
Same-input DEC=2 vs real stock, 43520 calls: worst-row rel median 4.70e-4, max 1.73e-3, nonfinite 0, no outliers; stock's own trajectory same distribution (max 1.59e-3).
Carry-forward: yes, within chunk; at eval 1848 Q rel l1 .019, l10 .692; K rel .075-.160, V rel .086-.207. Chunk 10 alone with DEC=2 gives 6.679251.
SPLIT: selects a chaotic realization, not a cause. SPLIT=3 + 2e-6 noise gives .334 and 6.300 at 10:416.
Amplifier, l1 Q response at call 336, q8_1 path vs no-q8 debug route: 1e-6 input 3.29e-3 vs 9.78e-7; 1e-4 input 3.48e-2 vs 5.74e-6.
VERDICT: amplification of rounding-level differences through q8_1 activation rounding, also present in stock; no DEC=2 kernel bug shown. Not determined: whether 10:416 stays low without q8_1 (chunk 0 only reached); >3 rate (1 event). Owner decides on gate 2 (still FAIL as written).

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

## Scout 3 (Fable, 2026-11-09)
Read-only scout, no GPU, no build. Base C4 (coordinator correction: owner adopted DEC=2 despite the gate FAIL). Spec: scout3-pick.md.
- PICK: existing GGML_CUDA_FP32MV (f32-activation mmvq, n=2..4) on C4. In the C4 lib already, no build. KLD never run on it.
- Measured basis: q8_1 removed on the debug route took the DEC=2 16-chunk mean .010614 -> .004487, max 6.679 -> 1.641 (not a clean pair). Op n=3 214 -> 194 us. K-fp32mv 7 Nov: tg1 -7%, tg2agg +6%, step flat, accept .754 -> .665 on one prompt with probe .476, so on diverged text.
- ESTIMATES against C4, unmeasured: KLD -30 to -60%, tg@52k/@100k 0 to +3%, short tg -7% to +1%, tg2agg 0 to +6%, about 50% chance of passing.
- First test: 16-chunk KLD, arms C4 vs C4 + FP32MV, full prod env, ~10 GPU min (estimate), no build.
- Criterion fixed now: d mean <= +.0005, same-top >= -.3 pt, F mean <= C mean - .0030, no nonfinite, no new token > 3.
- Owner ruling needed: limits are worded two-sided; a large KLD drop breaks the wording though it is better.
- Gaps found: FP32MV skips n=1, n>=5, ids and fusion, so 2-slot prod stays partly on q8_1 (Stage 2 build, Opus). f2/kldrun.sh set 3 GGML_CUDA flags while prod C4 sets 17, so the 64-chunk gate did not run on prod's env. NORM_Q8 stale-input risk with FP32MV not excluded.
- Runner-up: draft 3 at long ctx on C4 (no build, ~1 h, gain on filler text only, estimate).

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

## Power cap sweep, 9 Nov 2026 (partial, on C4; owner keeps 250 W)
Against 250 W: short tg 22.41, tg@52k 18.01, pp4k 91.7. Sweep stopped early; tg@100k not measured. Raw: power/sweep-20261109.jsonl
| cap | short tg | tg@52k | pp4k | energy short/52k/pp |
|---|---|---|---|---|
| 225 W | -3.6% | -3.6% | -4.6% | 0/-6/-5% |
| 200 W | -9.5% | -9.1% | -17.0% | -5/-9/-3% |
| 175 W | -24.6% | -20.5% | -30.7% | - |
| 150 W | -37.6% | -37.6% | -38.4% | - |
