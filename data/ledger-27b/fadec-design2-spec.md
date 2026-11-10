# fadec design 2: GQA-grouped decode FA for q4_0 KV (GGML_CUDA_FA_DEC=2) -- SPEC ONLY, nothing built
Why: design 1 (vec ncols=4) was 30-37% slower: vec runs one block per q-head, so each kv-head's K/V is streamed 6x. TILE+dequant wins today.

Selection code (fadec tree, ggml/src/ggml-cuda/fattn.cu):
- Generic sm_52 tail, ~lines 718-731: `if (can_use_vector_kernel) {... else { if (Q->ne[1] <= 2) return VEC; } } return TILE;` (Q rows 3 -> TILE; TILE sets need_f16_K/V in get_alloc_size ~line 750, which is the per-step q4_0->f16 dequant of the whole cache).
- Hook for design 2: in ggml_cuda_flash_attn_ext (~line 786), before the occamy block: `if (ggml_cuda_fa_dec_mode()==2 && ggml_cuda_fa_dec_applicable(dst)) { ggml_cuda_fa_dec(ctx,dst); return; }`. Selection untouched, so flag 0 and flag 1 stay byte-identical. Design-1 env_dec in fattn-common.cuh must test ==1 only.
- Applicable: Q f32, K=V=q4_0, D=256, mask f16 (ne[2]==1), no sinks/bias/softcap, G=Q->ne[2]/K->ne[2] in {4,6,8}, 1<=Q->ne[1]<=4, G*rows<=24, K->ne[1]%128==0, dst contiguous, Q->ne[1]>=GGML_CUDA_FA_DEC_MINQ (default 3).

Kernel (new fattn-dec.cu/.cuh; template<G,NR>, 128 threads, ~37 KB static smem, 2 blocks/SM):
- grid (nsplit, kv_head, seq). Block loads Q for all G*NR queries of its kv-head into smem (pre-scaled, permuted to float4 groups [lo2i,lo2i+1,hi2i,hi2i+1] matching q4_0 nibble order).
- Tile = 128 KV tokens, tiles interleaved across splits (tile = split; += nsplit). Mask loaded first; __syncthreads_or skips fully masked tiles (idle slot, padding).
- Phase A: thread = 1 token; reads its 144 B K row via ushort loads, unpacks nibbles once, FMAs against all NQ queries (LDS.128 broadcast, 4 FMA per LDS), per-32-block partial times fp16 scale in f32. S[NQ][128] to smem (+mask).
- Phase B: warp-per-query online softmax over the tile (max, exp, sum, alpha) -> P in smem, m/l/alpha in smem.
- Phase C: thread owns dims {b*32+j, b*32+16+j} (one q4_0 byte) of V; per token 1 byte + scale load, 2 dequants, NQ*2 FMA with P read as float4 over 4 tokens; acc[NQ][2] in registers rescaled by alpha per tile.
- Epilogue: partials part_o[blk][NQ][256], part_ml[blk][NQ]; small combine kernel (grid NQ x kvh x seq, 256 thr) does max/exp-weighted merge and writes dst[((seq*N+row)*H+head)*256+d]; guard L==0.
- nsplit = ceil(96/(kvh*nseq)) (env GGML_CUDA_FA_DEC_SPLIT to tune), capped at ntiles. Instantiate (G,NR): (6,1..4),(4,1..4),(8,1..3).

Files: ggml-cuda/fattn-dec.cu (new), fattn-dec.cuh (new), fattn.cu (hook + include), fattn-common.cuh (env_dec ==1). Backups *.bak-20261108-fadec exist for the three existing files. Build: build_b11552_fadec.sh (cmake reglobs); max 4 builds.

Traffic/compute per step at 100k ctx, 1 active slot, 16 attn layers, 4 kv heads, D=256:
- Today: dequant reads 0.5 B/elem + writes 2 B/elem for K and V, then TILE reads f16 (2 B/elem): ~4.5 B/elem x 2 x 100k x 4 x 256 x 16 = ~15 GB/step => ~50 ms at 317 GB/s (matches +70 ms growth).
- Design 2: reads q4_0 once (0.5625 B/elem) = ~1.8 GB/step => ~6 ms memory. Compute floor: 18 queries x 256 x 2 (K,V) FMA per token-kvhead = ~29 GFMA... ~20-25 ms at 60-70% of M6000 fp32 peak. Expect FA+dequant 70 ms -> ~25-30 ms: step 124 -> ~85 ms (tg@100k 8.07 -> ~11.5). Risk: reaching >50% FMA efficiency by hand; LDS broadcast pressure.

Test plan (all under flock gpu-bench.lock, no prod restart):
1. tbo FLASH_ATTN_EXT 3993/3993 for DEC=0/1/2 (+ GGML_CUDA_FA_DEC_DEBUG counter to prove the new path ran; check tbo has G=4/8 q4_0 hsk256 cases since G=6 may be absent).
2. KLD: flag-0 control at `-c 512 --chunks 16 -b 3 -ub 3`, then flag 2 same; gate |dmean KLD|<=0.0005, same-top within 0.3 pt; plus standard c512 (mean ~0).
3. tg_long_fadec.py flag 0 vs 2 at reps 0/2600/4900, 2 runs each; also MINQ=1 variant for rows 1-2. Candidate if tg@100k >=+5%, short tg >=-0.6%, long pp >=-0.6%.
