# Plan: bare-tensor launch surface (remove StridedTileArg's runtime plumbing)

Goal: a tile kernel's launch surface becomes **bare tensors + comptime
knobs**; the `Tile` is constructed in-kernel on the first line, exactly where
`arg.tile()` runs today. **Hard rule: performance same or better everywhere,
proven per step — a step that regresses any gate does not merge.**

## Why (evidence, 2026-07-31, metabolic attention work)

- `metabolic-extension/src/attention_tile/kernel.rs::attention_raw_kernel`
  (env `ATTN_TILE_RAW=1`) is the parity reference: the tile-DSL attention's
  exact math launched with bare tensors + `launch_unchecked` + in-kernel
  offsets. It measures at **full production parity** (47.0/48.4 vs auto
  47.8/47.8 tok/s, qwen3-1.7b decode); the identical kernel through
  StridedTileArgs measures ~35µs/call slower (−4.7% model tok/s). The arg
  machinery is the whole residual.
- The arg's contents split three ways:
  1. **comptime** (Space, Storage, check, StagePlan, vector width, quant
     scheme) — already free, baked into the JIT key; keep as comptime kernel
     params.
  2. **runtime duplicating the tensor's own metadata** (window bounds ≈
     shape; physical strides ≈ strides ÷ line width; window_start = 0 at
     launch) — reconstruct in-kernel from the `Tensor` metadata cubecl
     uploads anyway.
  3. **genuinely extra**: quant scales (a real second tensor — pass it as
     one) and TMA tensor maps (keep `TmaTileArg` unchanged).

## Steps

0. **Baselines before touching anything.** Record, with the
   alternating-isolated-process discipline (never A/B in one process; pair
   ratios; the M2 Pro throttles):
   - cubek matmul benches (`cargo bench -p benchmarks --bench ...`),
   - metabolic gemm anchor (`scripts/gemm_anchor.py` +
     `tests/gemm_codegen.rs::perf_sample`),
   - metabolic attention model A/B (`BENCH_ATTENTION=custom-kernel-tiled`
     vs unset, `cargo bench --bench model --features metal ...`, 3 pairs).
   These are the perf gates for every step below.

1. **`Tile::of` (new construction, cubek-tile).**
   `Tile::of::<E, W>(t: &Tensor<Vector<E, W>>, #[comptime] space: Space,
   #[comptime] storage: Storage, #[comptime] check: bool, ...)` building the
   same `MemData` `from_concrete`/the launched path builds today, but off
   the tensor's shape/strides (mirror the constructor in
   `tile/mem.rs` ~line 146). Unit-test it against the existing path: same
   space + same tensor ⇒ bit-identical reads/writes.

2. **Comptime axis relabels.** The host-side stride surgery (metabolic's GQA
   (B,H,1,D)→(B,KV,G,QP,D)) becomes a comptime axis-split spec consumed by
   `Tile::of` (split one tensor dim into (outer, inner) with a comptime
   inner extent). Keep v1 minimal: exact-rank plus splits; merges later.

3. **Broadcast-by-omission** needs nothing: which axes an operand omits is
   the comptime Space projection, untouched.

4. **Quant.** A quantized operand = values tensor + scales tensor + comptime
   scheme; build the `QuantInfo` in-kernel (mirror `QuantInfo::native`).
   Mind the served-vs-physical width split (packed-u32).

5. **Migrate behind the existing seam.** Kernels already construct through
   `DeliveryFamily::tile()` — change what `D::Arg` is under the trait, one
   routine at a time (matmul register → matmul cmma → attention), leaving
   the old arg path alive until each routine passes its perf gate. Delete
   the old path only when nothing uses it.

6. **The `Launcher` keeps its host jobs** (cube geometry, vector-width
   gating, overhang→check decision) but ships nothing — its `arg()` yields
   the bare tensor + the comptime bundle.

## Dangers

- **JIT-fork granularity**: anything moved runtime→comptime forks a compiled
  kernel per value. Truly dynamic sizes must stay `Extent::Dynamic`, read
  off the tensor.
- The overhang-derived `check` decision must reproduce exactly (ragged
  shapes; a checked operand cannot vectorize).
- CPU runtime parity (tests run there too).
- Fusion stays unsupported for tile kernels (needs virtual tensors) —
  unchanged, don't chase fusion-only failures.
- TMA untouched.

## Expected payoff

~35µs/launch on the M2 Pro per tile kernel call (measured on attention:
the full remaining −4.7% model tok/s), plus whatever the gemm path pays
today; smaller per-dispatch metadata for every tile-DSL kernel.
