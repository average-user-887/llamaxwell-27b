# Qwen3.8-27B community weights: candidates to check (9 Nov 2026)

Goal: fewer thinking tokens without large quality loss. Nothing downloaded or run yet.

Method: two local research cards (35B via Firecrawl) found the leads. Claude then checked them against the HF API, GitHub API, model cards and GGUF tensor tables (first 45 MB of each file).
Downloads are HF's last-30-days figure.
**No independent reproduction exists for either finetune.** All accuracy and token numbers below are author-run. modelfit.io only recomputed the published tables and says it "found no independent reproduction of either".

## Test these

| # | Weights | Claimed thinking cut | Claimed accuracy (base → finetune) | 30-day dl / likes | License | Getting our Q4_0 |
|---|---|---|---|---|---|---|
| 1 | [ThinkingCap-Qwen3.8-27B](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B) ([GGUF](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF)), 23 Sep | 37.2% mean of per-benchmark %; **22.8% by tokens** (15,735→12,144) | 86.65→85.79 over 12 benchmarks; range +2.25 pp (AA-LCR) to −3.86 pp (AIME 2026) | GGUF 922,733 / 96 | PolyForm Small Business + personal-use grant | Requant from f16 GGUF (54.7 GB) |
| 2 | [Swift-1.5-Qwen3.8-27B](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B) ([GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GGUF)), 16 Sep | Mean 16–45% per benchmark (GPQA 41.9%); headline 58.5% is **GPQA median only**; 29.0% by tokens (modelfit's recomputation) | GPQA 88.28→88.59, Terminal-Bench 69.21→72.13, LiveCodeBench 76.76→81.71, C-Eval 90.00→90.92; AIME 98.67→96.00, HMMT Nov 99.33→97.33, IFBench 73.53→72.07, ERQA 67.45→65.40 | GGUF 125,888 / 260 | Swift Open License: free under US$1M revenue | No F16 GGUF or Q4_0: convert safetensors (55.6 GB) |
| 3 | [Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b) ([GGUF](https://huggingface.co/ukisai/Swift-Qwen3.8-27B-GGUF)) 1.0, 8 Sep | Mean 41.0% on GPQA; headline 58.3% is GPQA median | GPQA 88.38→88.28, AIME 98.67→94.00, HMMT Nov 99.33→96.00, LiveCodeBench 76.76→81.55 | GGUF 479,122 / 477 | Same as Swift 1.5 | **Ready-made Q4_0, 16.0 GB**: cheapest smoke test, but superseded by 1.5 |

Reading these numbers:
- Compare the cards with care: base LiveCodeBench is 91.14 on ThinkingCap and 76.76 on Swift, so the harnesses differ. Swift's coding "gains" may come from the base being cut off by a token limit (my inference; a blog also suggests it).
- Ignore the speed claims. Swift 1.5 claims "9.18×" and Swift 1.0 "×1.95"; neither is a typical speedup we should expect.
- Math is the common loser on all three.

Fit with our M6000 setup:
- **MTP works in principle.** All three carry the 15-tensor `blk.64` head (866 tensors, same as prod; a non-MTP file has 851). Swift 1.0's card says this is the base model's head, so draft acceptance on retuned weights may fall. Measure it.
- **Their Q4_K_M won't do.** It is 17.4 GB against our 15.4 GB head4 and doesn't have the Q4_0 layout we tuned for. Build Q4_0-head4 instead.
- **The KLD gate doesn't apply.** These weights differ on purpose, so KLD against stock proves nothing. They need an answer-quality test.

## Free first test: settings on stock 27B

| Setting | Evidence | Cost |
|---|---|---|
| `reasoning_effort: medium` (`chat_template_kwargs`) | BottleCap blog: stock tokens −52.1%; ThinkingCap at medium −60.2% | **Accuracy −9.16 pp stock, −9.90 pp ThinkingCap**: far worse than the finetunes at default effort |
| `enable_thinking: false` | [Simon Willison](https://simonwillison.net/2026/Aug/16/qwen-38-27b/), one SVG prompt on a Mac: 21 min / 22,276 reasoning tokens at xhigh vs 137 s / 3,715 total tokens off | Quality not measured; a sample of one |
| Per-task budget in our harness: [Yunado/dsh-qwen38-local-qol](https://github.com/Yunado/dsh-qwen38-local-qol) (25 stars, MIT, pushed 8 Nov) | Per-request thinking budgets, compaction that stops burning the output cap on thinking (repo description only; README not read, code not audited) | Audit before installing |

Takeaway: use low effort or thinking off for chores like titles and compaction, and keep thinking on for hard work. A finetune looks like the better trade for general use.

Lesser leads: [zhubaohi/dsh-qwen38-compaction-fix](https://github.com/zhubaohi/dsh-qwen38-compaction-fix) (4 stars, MIT, compaction only). [soster/qwen38-thinking-levels](https://github.com/soster/qwen38-thinking-levels) (24 stars) configures the `pi` client, so it is a reference at most.

## Popular, but not for this goal

| Weights | 30-day dl / likes | Why skip |
|---|---|---|
| [DavidAU TURBO Cold Fusion Heretic MTP](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF) | 1,530,688 / 2,738 | Uncensored multi-stage merge; research found no thinking-reduction figure or benchmarks (page itself not read) |
| [ISTA-DASLab GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | 1,356,774 / 2,598 | Quantization only, thinking unchanged. Worth a separate look for size: IQ3_S with MTP is 12.1 GB |
| [ThinkingCap RCO 4.2bpw](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF-RCO-4.2bpw) | 7,972 / 13 | Same weights as #1, smaller at 14.3 GB; no KLD or speed data for our kernels. Only if #1 wins |
| [prism-ml Ternary Bonsai 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) | 273,928 / 1,442 | Tagged as base Qwen3.6, not 3.8; 2-bit quantization, no thinking claim found |
| [tencent DSpine b16](https://huggingface.co/tencent/Qwen3.8-27B-DSpine-b16) | 1,464 / 134 | Speed drafter, not a thinking change. Safetensors only, created 7 Nov. Watch for GGUF and llama.cpp support |

## Proposed test (when you say go)

1. Download to the Models disk (164 GB free): ThinkingCap f16 and Swift 1.5 safetensors, about 110 GB, then build Q4_0-head4 of each. Remove the sources only with your OK.
2. Run stock vs ThinkingCap vs Swift 1.5 at the default effort on one short fixed prompt set from our real work, plus a small holdout with checkable answers. Add stock at `medium` as a fourth arm.
3. Measure thinking tokens, answer tokens, wall time, MTP acceptance, and accuracy.
4. Swap in the winner only if thinking drops by at least 20% and it loses no more than about 1 pp, or fails no real task stock passes. These bars are my proposal.

## Gaps

- Reddit (login wall) and ModelScope (JS shell) couldn't be read, so community opinion there is missing.
- License texts weren't read in full. Both look fine for a personal lab; check before any business use.
- The local cards mislabelled some evidence: Swift's numbers were called an "independent test", and modelfit's recomputation an "independent audit". Both corrected here.
