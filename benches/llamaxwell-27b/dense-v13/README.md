# Dense v13 serving benchmark

Dense v13 improved warm generation by **4.03%** and reduced request wall time by **2.97%** over C6 at a 77,920-token saved prefix. On the 53-token cold prompt, prefill improved **5.90%** and request wall time fell **5.09%**. These are medians of paired changes on a Quadro M6000 24 GiB (`sm_52`) using Qwen3.8-27B-Q4_0-head4 with MTP draft depth 2.

| Workload | Run order | Pairs | Prefill | Client generation | Server generation | TTFT | Request wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 53-token cold input | C6 → v13 | 1 | +6.20% | | | −5.81% | −5.24% |
| 53-token cold input | v13 → C6 | 1 | +5.60% | | | −5.30% | −4.94% |
| 77,920-token saved prefix | C6 → v13 | 3 | | +3.54% | +3.32% | +0.98% | −2.92% |
| 77,920-token saved prefix | v13 → C6 | 3 | | +4.43% | +4.29% | −0.27% | −3.52% |

The four fresh runtime instances followed C6, v13, v13, C6. Each emitted 16 tokens on the cold prompt and ran three timed 128-token generation requests plus one stop-word request against the same restored prefix. The deep requests reused 77,916 tokens, replayed four and selected a padded attention width of 78,080. Client generation uses the stream interval between output tokens 8 and 120. The observed v13 dispatches included the 49-column gate tile, six-split N1 direct-V draft attention and four-split N3 draft attention.

All 20 greedy request trajectories matched, including 433 generated token IDs per arm and 1,732 across all four arms. The comparisons also matched streamed text, prompt and cache counts, MTP proposals and acceptances, output count, finish reason, truncation and the stop word on deep requests.

One of six timed deep pairs had matching sampled P0 clocks (1,126 MHz graphics and 3,304 MHz memory). In that pair, client generation improved 4.43% and wall time fell 3.80%. Neither short pair had matched clock samples. The remaining timings include clock variation or missing samples. The parity result covers these greedy fixtures. Logprob distributions and performance under concurrent requests were unmeasured.

## N3/P4 lower boundary

On the same M6000, an OFF, ON, ON, OFF replay isolated `GGML_CUDA_FA_DRAFT_F16_P4=0/1` with the same v13 CUDA library and profile. The target prompt had 65,265 tokens, of which 65,261 were reused and four replayed. The draft attention physically dispatched at 49,152 padded K/V cells, the first enabled width. The two ON arms both selected the four-warp kernel on the stock 1 x 4 x 12 grid. OFF used eight warps.

| Run order | Timed pairs | Client generation gain | Server generation gain | Request wall time saved |
| --- | ---: | ---: | ---: | ---: |
| OFF to ON | 3 | 0.468% | 0.463% | 0.403% |
| ON to OFF | 3 | 0.434% | 0.394% | 0.336% |
| Pooled | 6 | 0.451% | 0.428% | 0.400% |

Each arm ran one dispatch witness and three timed warm requests of 128 output tokens. All 512 generated IDs per arm matched, along with text, cache and prompt counts, MTP proposals and acceptances, output count, stop behavior, truncation and slot ID.

One pair per order had a shared P0 sample at 1,101 MHz graphics and 3,304 MHz memory, with one sample per arm in each pair. Client generation gains in those two pairs were 0.387% and 0.425%. The other four pairs lacked a shared P0 observation. The [boundary JSON receipt](n3-p4-49152.json) records absolute rates, time to first token and wall times for each pair. These results cover this warm cache fixture.

## Measured profile

Both profiles used two 131,072-token slots, separate Q4_0 K/V caches, flash attention, batch size 2,048 and microbatch size 1,024. The measurement pins the same model, server and non-CUDA libraries across arms. Dense v13 used the CUDA library from source commit `536ce1093a4cb4eb76f6e1f69c39aca832e3d731` with 17 added switches. The [JSON receipt](receipts.json) contains all switch values, binary hashes and individual paired timing rows.

With the pinned runtime and v13 CUDA library already selected, run this Bash example from the receipt directory. Set `MODEL_GGUF` to the model file and `SLOT_DIR` to a slot save directory.

```bash
mapfile -t profile_env < <(
  jq -r '(.profile.shared_env + .profile.v13_added_env)
         | to_entries[] | "\(.key)=\(.value)"' receipts.json
)
env "${profile_env[@]}" llama-server -m "$MODEL_GGUF" -ngl 999 --fit off \
  -np 2 -no-kvu -c 262144 -ctk q4_0 -ctv q4_0 -b 2048 -ub 1024 -fa on \
  --cache-ram 4096 --ctx-checkpoints 8 --checkpoint-min-step 2048 \
  --spec-type draft-mtp --spec-draft-n-max 2 --no-mmproj --jinja \
  --slot-save-path "$SLOT_DIR"
```
