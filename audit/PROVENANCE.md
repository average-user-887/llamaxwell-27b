# Provenance - Qwen3.8-27B build (as of 2026-11-10)

## What runs
Container `llamacpp-m6000-27b` = image `local/llama.cpp:server-cuda-b11552-sm52-normq8` (built 2026-10-12, shared with the 35B) with **`/app/libggml-cuda.so` bind-mounted over it**.

| Piece | Source | Identity |
|---|---|---|
| `llama-server`, `libllama*`, `libggml-base/cpu` | image, source = commit 2 (`normq8`) | `image-app-sha256.txt`, `image-inspect-slim.json` |
| `libggml-cuda.so` (deployed) | commit 4 (`fadec2 as built`) | **sha256 `7f81976eb30ce79727fa02690b099c2bd105f3674bef6383b2e7b3fa8f2e2238`**; release asset `libggml-cuda-27b-fadec2-7f81976e.so` |
| model | `Qwen3.8-27B-Q4_0-head4.gguf` (not included) | flags in `server-args.txt`, mounts in `mounts.txt`, launcher `run-solo-27b-mtp-20261109-c4.sh` |

## Commit ladder
1. `23b0202` upstream b11552.
2. normq8 - image's server/libllama source.
3. gate-copy tree - Occamy kernels.
4. **fadec2 as built** - gate-copy plus the 2026-11 work (mmvq tuning, fagemm, `fattn-dec` GQA q4_0 decode, `GGML_CUDA_FA_DEC=2`), folded into one commit. *Reconstructed*, see below.
5. post-build working-tree edits (fadec2dbg experiments, 8 Nov 14:44-18:07). **Not in the deployed library.**

## Build recipe
`audit/build_b11552_fadec2.sh` + `audit/build-b11552-fadec2-20261108T132831.log`: `nvidia/cuda:12.8.1-devel-ubuntu24.04`, gcc-14, `-DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DCMAKE_CUDA_ARCHITECTURES=52-real -DCMAKE_BUILD_TYPE=Release`, `mmvq_tune.h` written by the script (values `1 2 2 2 4 4 1 0`). The script builds a *copy* of the tree mounted at the old path `/w/src-b11552-mmvqtune-20261107` because the CMake cache is path-bound.

## Known discrepancies (read before auditing)
- **Source tree was edited after the deployed build.** The library was built 2026-11-08 13:29; the working tree was modified afterwards (14:44, 16:36, 18:06). Commit 4 restores the as-built state from the `.bak-20261108-fadec2dbg` copies (`fattn.cu`, `fattn-dec.cu`, `fattn-dec.cuh`, `ggml-cuda.cu`) and regenerates `mmvq_tune.h`; **no snapshot was taken at build time**, so this is the best reconstruction, not proof. `ggml-cuda.cu` is the least certain file (its backup was taken at 18:06).
- **Not verified by rebuild.** No rebuild from commit 4 has been compared to the deployed hash. A matching rebuild is the check that would confirm the reconstruction.
- The file was installed as `prod-libs/libggml-cuda-fadec2-20261109.so`; the builder wrote `mmvqtune-libs/libggml-cuda-mmvqT-fagemm-fadec2.so` (identical bytes, same sha256).
- No D14 `norm.cu` change here (that is 35B only).
- Receipts/paths refer to `/media/avg-usr/Storage/m6000-llama-cpp/`.

## Verify
```
git log --oneline
git diff 23b0202 HEAD~1 -- . ':!audit'     # deployed-build delta vs upstream
git diff HEAD~1 HEAD -- .                  # post-build edits (not deployed)
sha256sum libggml-cuda-27b-fadec2-7f81976e.so   # from the release
```
