# Plan: bare-tensor launch surface (retire VecTensor + the TileArg wrapper)

Goal: **get rid of TileArg.** A tile kernel receives what everything else
in metabolic receives — **tensors** (runtime data) **+ blueprint-derived
comptime decisions** — with `Tensor<E>`, `E: CubePrimitive<Scalar: Numeric>`
(`Vector<F, N>` for a lined operand, a scalar type otherwise), and the
first line in the kernel is the **Tensor → Tile** transformation
(`Tile::of`), replacing today's TileArg → Tile (`arg.tile()`). TileArg is
a third thing between those two: it re-bundles the tensor with decisions
the blueprint already owns, and its per-dispatch construction costs host
time. Both the overhead and the duplication go away with it. **Hard rule:
performance same or better everywhere, proven per step — a step that
regresses any gate does not merge.**

## Why (evidence 2026-07-31, revised 2026-08-03)

- `metabolic-Paul` `attention_tile/kernel.rs::attention_raw_kernel`
  (env `ATTN_TILE_RAW=1`, branch `feat/attention/tiled`) is the evidence:
  the tile-DSL attention's exact math launched with bare
  `Tensor<Vector<F, N>>` + `launch_unchecked` + in-kernel offsets measures
  at **full production parity** (47.0/48.4 vs auto 47.8/47.8 tok/s,
  qwen3-1.7b decode); the same math through `StridedTileArg` measures
  ~35µs/call slower (−4.7% model tok/s). Caveat: the raw arm also changed
  the split ending, so the 35µs is an upper bound on the arg surface —
  step 0's overhead probe re-isolates it before we attribute all of it.
- What is **already true at HEAD** (post #427/#448): `StridedTileArg` is
  thin — tensor + comptime space/storage/vector_size + quant option — and
  `MemData::from_tensor` already rebuilds shape/strides/window in-kernel
  from the tensor's own metadata. The residual to remove:
  1. **`VecTensor`** — custom binding machinery (launch-time width, `Deref`
     expand hack, its own IR metadata plumbing) standing in for a plain
     tensor whose element type carries the width.
  2. **The `CubeLaunch` arg-struct wrapper** per operand
     (`StridedTileArg`, `QuantArg`) and its per-dispatch
     construction/upload on the host.
  3. **Quant scales ride inside `QuantArg` as an `OwnedTensor`** — they
     become an ordinary second tensor parameter.
  - Comptime knobs (Space, Storage, check, StagePlan, quant scheme) are
    already free — they move from arg fields to comptime kernel params.
  - TMA (`TmaTileArg`) untouched.

## Steps

0. **Baselines before touching anything.** Alternating-isolated-process
   discipline (never A/B in one process; pair ratios; the M2 Pro
   throttles):
   - cubek: `cargo bench -p benchmarks --bench gemm` / `gemv` /
     `quantized_matmul` (metal),
   - metabolic-George: `cargo bench --bench model --features metal` +
     `--bench gemm`, 3 pairs,
   - **new: a launch-overhead probe** — a decode-shaped tiny dispatch
     (e.g. the gemv tiled path) timed over many calls, so the per-call
     arg cost is observed directly in this repo. (The attention parity
     rig lives only on metabolic-Paul's branch; it is not a gate here.)
   These are the perf gates for every step below.

1. **`Tile::of` (new construction, cubek-tile).**
   `Tile::of::<E>(t: &Tensor<E>, #[comptime] space: Space,
   #[comptime] storage: Storage, #[comptime] check: bool, ...)` with
   `E: CubePrimitive<Scalar: Numeric>`; the width is read comptime off `E`
   (scalar + lane count), no launch-time width. Builds the same `MemData`
   that `from_tensor` (tile/mem.rs:101) builds today, minus the VecTensor
   detour. Verify the metadata units first: with a line-typed binding,
   whether `shape`/`stride` come back scalar-unit or line-unit decides the
   ÷w fixups — confirm on Metal **and** the cpu runtime. Unit-test against
   the existing path: same space + same tensor ⇒ bit-identical
   reads/writes.

2. **Comptime axis relabels.** Host-side stride surgery (e.g. the GQA
   (B,H,1,D)→(B,KV,G,QP,D) relabel on Paul's branch; batch splits here)
   becomes a comptime axis-split spec consumed by `Tile::of` (split one
   tensor dim into (outer, inner) with a comptime inner extent). Keep v1
   minimal: exact-rank plus splits; merges later.

3. **Broadcast-by-omission** needs nothing: which axes an operand omits is
   the comptime Space projection, untouched.

4. **Quant.** A quantized operand = values `Tensor<EStore>` + scales
   `Tensor<f32>` + comptime scheme; build the `QuantInfo` in-kernel
   (mirror `QuantInfo::native`, tile/mod.rs:122). Mind the
   served-vs-physical width split (packed-u32: the binding is u32-typed,
   the served width is ×pack).

5. **Migrate behind the existing seam.** Kernels already construct through
   `DeliveryFamily::tile()` — change what `D::Arg` is under the trait, one
   routine at a time, leaving the old arg path alive until each routine
   passes its perf gate:
   - cubek: cubek-matmul cmma, cpu_gemm, cubek-quant dequantize_tiled,
     the eval benchmarks;
   - metabolic-George: `gemm/launch.rs`, `gemv/tiled/launch.rs` (+ the
     gemv_v2_spike rig).
   Delete `StridedTileArg` + `VecTensor` only when nothing uses them.

6. **The `Launcher` keeps its host jobs** (cube geometry, vector-width
   gating, overhang→check decision) but ships nothing — its `arg()` yields
   the bare tensors + the comptime bundle.

## Step 0 results (2026-08-03, M2 Pro)

- **Launch-overhead probe** (`metabolic-George
  tests/launch_overhead.rs`, tiny 1×256×1024 f16 gemv, 3 operands,
  medians of 9×200-launch windows): tile-arg surface **9.6–10.2µs/call
  host enqueue vs 6.3–7.0µs bare — a ~3–4µs/call delta, ≈1.2µs per
  operand**. The synced mode can't isolate the surface until the same
  kernel runs both ways (step 5); the probe's two bodies differ GPU-side.
  Cross-check: both arms compute the same gemv and agree (<2e-2 rel).
- Reading: the −4.7% attention tok/s is consistent with ~3-4µs × the
  hundreds of tile dispatches a decode step issues, host-bound — the
  payoff is real but per-operand, not a 35µs lump.
- Metadata-units question (step 1) largely answered from source:
  `VecTensor` doc + Paul's raw kernel show a line-typed binding keeps
  scalar-unit shape/strides with line-granular indexing — the same
  contract `MemData::from_tensor`'s ÷w fixups already implement. Residual
  check: `vector_size()` and the slice re-type on `Tensor<Vector<F, V>>`.

## Step 1 results (2026-08-03)

- `Tile::of` landed in `cubek-tile/src/tile/mem.rs`:
  `Tile::of::<E: CubePrimitive<Scalar = T>>(t: &Tensor<E>, #[comptime]
  space, #[comptime] storage)`. The width is the element type's
  (`tensor.vector_size()` reads it comptime off the binding type); the body
  is `from_tensor`'s scalar-unit → line-unit conversion minus the VecTensor
  detour; the slice re-type is the same static coercion.
- Parity proven bit-for-bit (`tests/tile/bare.rs`): the identical staged
  mma through `StridedTileArg::tile` vs `Tile::of`, width 1 and width 4
  (`Vector<f32, Const<4>>`), exact-equal outputs on **wgpu/Metal and the
  CPU runtime** (`--features cubecl/cpu`). Full cubek-tile suite green
  (113 tests).
- Launch-surface note for step 5: call sites pass **scalar-unit**
  `TensorArg`s; the width rides in the element type (comptime `Const<W>`
  or launch-resolved `Size` value, Paul's raw-kernel pattern). The
  test-utils `tensor_arg(w>1)` pre-divides shapes into line units — that
  contract is for raw slices, not for `Tile::of`.
- Seam consequence: `DeliveryFamily::tile` must grow comptime
  `space`/`storage` params — a bare tensor no longer carries them.
- cubek gemm baseline sweep was stopped partway (829/1356 rows recorded,
  through 6144³; the 8192³ tail is missing) — sufficient as an anchor;
  step-5 gates are paired A/B runs, not comparisons to this file.

## Step 4 + step 5 (gemv) results (2026-08-03)

- `Tile::of_dequant` (values tensor + plain scales tensor + comptime
  scheme, `QuantInfo` built in-kernel) — parity bit-exact vs
  `tile_dequant` on packed-u32 block Q8S, Metal **and** CPU
  (`tests/tile/bare.rs`, 3 tests).
- **gemv migrated** (metabolic `gemv/tiled/launch.rs`):
  `gemv_bare_quant_kernel` / `gemv_bare_float_kernel` — bare tensors +
  comptime spaces/storage/scheme, widths as `Size` generics fed launch
  values, scales as an ordinary 4th tensor. Old path intact behind
  `GEMV_TILEARG_BARE` (default off).
- Correctness: full metabolic correctness suite through the bare surface —
  all green except 2 **pre-existing** failures
  (`gemv_quantized_is_deterministic`, `gemv_quantized_ignores_pool_garbage`,
  both the *shared-memory* gemv, failing identically at pristine HEAD on
  the git-pinned cubek — not this work's).
- Perf gate (`tests/gemv_tilearg_ab.rs`, FFN decode shape 1×4096·4096×12288
  q8s, 8 alternating-process pairs): ratio median **1.006, IQR ~0.87–1.06 —
  parity**; the surface's ~3µs enqueue win is invisible under ±15% machine
  noise at ~70µs/call. Gate: **pass** (no regression).
- Wiring note: metabolic-George Cargo.toml carries a DO-NOT-COMMIT path
  override onto ../cubek-George; restore the git pins before any commit.
- Step 2 (axis splits) confirmed unnecessary for George's routines.
- **Step 6 landed with the register migration**: cubek's
  `StridedTileSource` refactored into a shared `realize()` +
  `build()`/`build_bare()`; `build_bare` yields `BareStridedSource`
  (plain `TensorArg` + width + projected `Space` + `Storage` + optional
  (scales, scheme)) — the Launcher keeps every host job, ships no wrapper.
  cubek-tile suite green (114).
- **gemm register migrated** (`gemm_register_bare_kernel`, behind the now
  shared `TILEARG_BARE` switch): correctness green incl. the
  register-composition rungs (32,40,48 / 33,35,37), batched-broadcast, and
  pool-garbage tests. Full metabolic suite under `TILEARG_BARE=1`: green
  except the two pre-existing shared-memory gemv failures.
- **Migration complete (2026-08-03), all behind `TILEARG_BARE`:**
  metabolic gemm cmma-strided (`gemm_cmma_bare_kernel` over the new
  `BareDelivery` trait — `Arg<E, V>` + seam-passed comptime space/storage;
  `BareStrided` = plain `Tensor<Vector<E, V>>`, Tma impl exists but
  metabolic keeps TMA on the wrapper path); cubek-matmul `cmma_bare_kernel`
  + `cpu_gemm_bare_kernel` (32 extended cpu_gemm tests green on the CPU
  runtime); cubek-quant `dequantize_bare`. `bare_surface()` lives in
  cubek-tile, one `TILEARG_BARE` flag for everything. Kernel-compile logs
  confirmed both bare gemm kernels actually run (15 cmma + 6 register
  compiles in one metabolic test sweep). Default (arg) path re-verified
  green across both repos after all edits.

## Cleanup pass (2026-08-03, contributor-book review)

- `TileSpec { space, storage }`: the comptime half of a bare operand as one
  named value ("pass the enum, not its ashes") — threaded through
  `Tile::of`/`of_dequant`, `BareDelivery::tile`, `BareStridedSource`, and
  every bare kernel (two comptime params per operand → one).
  `TileSpec::from_concrete` is the one derivation both launch surfaces
  share (`from_concrete` and `build_bare` both call it).
- `Launcher::bare_arg` replaces the repeated `.arg::<f32>` phantom-type
  annotation; `bare_surface()` exists once (cubek-tile; metabolic
  re-exports). Plan references and stale numbers stripped from comments;
  fmt + clippy clean; suites green both arms.
- `of_impl` transiently duplicates `from_tensor_quant` — the copy dies
  with `VecTensor` at the deletion step, like the twin kernels.

## Proof + deletion (2026-08-04)

- **Kernel-identity proof (Metal, `CUBECL_DEBUG_LOG`)**: the same cmma matmul
  compiled through both surfaces produced **instruction-identical kernels**
  (identical modulo SSA numbering and the declaration order of four
  equal-size smem buffers). The diff first caught a real divergence: the
  bare arm compiled with `stage.layout: Strided` where the wrapper had
  `Tiled` -- `strided()` owned the `StageStorage::for_space` default and
  `TileSpec::from_concrete` didn't. Fixed by moving the default into
  `TileSpec::new`, the one derivation every path shares. `dequantize`
  likewise identical modulo names; the CPU runtime doesn't dump source
  (bare arm verified running `CpuGemmBareKernel`, suites green).
- **Perf gates (M2 Pro, paired alternating processes + in-process
  interleaved A/B)**: Metal gemm cmma at 4096-cube/skinny/512-cube shapes:
  pair-ratio medians 1.0002 / 1.0001 / 1.0023 -- parity (in-process runs
  show a ~1% arm-order thermal skew; kernels are proven identical, so it is
  the protocol, not the surface). CPU cpu_gemm: process-pair median 1.021
  forward vs 1.006 reversed (order bias); in-process interleaved median
  **0.9825 -- bare faster**; invariant-strategy control confirms a ~1%
  noise floor. Gate: pass.
- **Deletion done**: `StridedTileArg`, `StridedTileArgLaunch`, `QuantArg`,
  `VecTensor`/`VecTensorArg`, the old `DeliveryFamily`, the twin
  `*_bare_kernel`s, `bare_surface()`/`TILEARG_BARE`, `Launcher::bare_arg`
  (and its phantom `f32`), and `MemData::from_tensor`/`from_tensor_quant`
  are gone. `BareDelivery` is now `DeliveryFamily` (V-typed `Arg`,
  seam-passed `TileSpec`); `BareStrided` is `Strided`; `build_bare` is
  `build`, yielding `StridedOperand` (with `bound_width()` owning the
  packed-width rule); cmma TMA rides the same kernel via the `Tma` family.
  Scheme validation moved into `build()` so the bare path gates quant at
  launch like the wrapper did. Tests migrated to `Tile::of` + `TileSpec`
  (`TileInput::spec()`); the bare-vs-arg parity tests died with their
  comparison target.

## One-space review fix (2026-08-04)

- Review: per-operand specs each carried their own pre-projected `Space` --
  three copies of the space in the JIT key, projections done host-side, and
  the copies able to drift or mismatch. Reworked to the one-space model:
  the kernel takes ONE comptime `Space` (extents + partitioning, single
  source of truth) and `TileSpec` shrank to `{ axes, storage, stage:
  Option<StageStorage> }` -- only what is per-operand. `Tile::of(tensor,
  space, spec)` does `space.project(&spec.axes)` in-kernel; the stage
  layout derives from the one space's leaf there too (the `TileSpec::new`
  stamping is gone; `staged()` is the explicit override). `Storage.stage`
  collapsed to `units`; `TmaTileArg` lost its own comptime space (the seam
  provides it).
- Proven codegen-neutral against the pre-change build: dequantize
  byte-identical, cmma identical modulo the benign equal-size smem
  allocation permutation.

## TileArg carrier (2026-08-04)

- Review follow-up: reintroduced a per-operand carrier, `TileArg<'a, E, V:
  Size> { tensor: &Tensor<Vector<E, V>>, #[comptime] spec: TileSpec }`,
  with `tile(space)` / `tile_dequant(scales, scheme, space)` methods --
  the tensor and its spec ship as one launch argument, so a tensor can
  never pair with another operand's spec and kernel signatures shrink
  (cmma: 13 params -> 9). NOT the old wrapper: no space copy, no
  VecTensor, no quant payload (scales stay a loose tensor; the discipline
  is tensor + spec, nothing else, ever). `TmaTileArg` symmetrically
  absorbed its spec; the `DeliveryFamily` seam is `tile(arg, space)`.
  `StridedOperand::arg()` produces the launch value (panics if the quant
  side-channel wasn't destructured first).
- Codegen-neutrality proven again by kernel diff: dequantize
  byte-identical, cmma identical modulo the equal-size smem allocation
  permutation. The CubeLaunch derive handles the `V: Size` generic
  (default type params don't work -- the derive appends its own generics).

## QuantOperand typestate (2026-08-04)

- Review follow-up: the quant destructuring dance (`mut` + `bound_width()`
  before `quant.take().unwrap()` before `.arg()`, guarded by a runtime
  panic) existed because quant-ness was an `Option` on one product type.
  The builder now carries a third typestate marker `Q`: `.quantized()`
  flips it, and `build()` returns `StridedOperand` (plain; no quant field,
  no assert) or `QuantOperand` (tensor + spec + scales + scheme as
  first-class fields, `bound_width()`, `arg()` yielding the TileArg plus
  the loose scales). Mispairing and silent scale-dropping are now
  unrepresentable; the ordering is enforced by the borrow checker, not a
  panic.

## QuantTileArg carrier (2026-08-04)

- Review follow-up: the residual two-line quant destructure existed only
  because scales/scheme lived outside the arg. A quantized operand now
  rides its own carrier, `QuantTileArg<'a, E, V: Size>{values, scales,
  comptime spec, comptime scheme}`, served by `tile::<O>(space)` -- a
  distinct total type (name-the-states), NOT the old ComptimeOption inside
  one arg type. `TileArg::tile_dequant` deleted; quant kernels lost their
  loose scales + scheme params; `QuantOperand::arg()` returns the one
  carrier, so a quant launch site is `let vb = b.bound_width();` +
  `b.arg()`. `QuantizedTileInput::arg()` mirrors it in test-utils.

## Remaining

1. metabolic-George: repin cubek, flip its gemv/gemm launches onto the
   deleted-wrapper surface (its bare kernels already exist behind
   `GEMV_TILEARG_BARE`/`TILEARG_BARE`), drop the twin kernels, restore the
   git pins (the DO-NOT-COMMIT path override).
2. metabolic model-level A/B (3 pairs, llama-3.2-1b decode) on a quiet
   machine, as a belt-and-braces check on top of the kernel-identity proof.

## Dangers

- **Width-in-type forks a compiled kernel per width.** That is the point
  (the width becomes comptime), and real widths are few (1/2/4/8), but it
  is a new JIT-key axis; truly dynamic *sizes* must stay
  `Extent::Dynamic`, read off the tensor.
- **wgpu/naga silent zeros**: VecTensor existed because re-grouping a
  scalar binding in-kernel needs `memory_reinterpret` (CUDA/HIP only).
  `Tensor<Vector<F, N>>` keeps the binding typed at the launched width —
  preserve that invariant; never re-line in-kernel.
- The overhang-derived `check` decision must reproduce exactly (ragged
  shapes; a checked operand cannot vectorize).
- CPU runtime parity (tests run there too).
- Fusion stays unsupported for tile kernels (needs virtual tensors) —
  unchanged, don't chase fusion-only failures.
- TMA untouched.

## Expected payoff

Up to ~35µs/launch on the M2 Pro per tile kernel call (attention
measurement; step 0's probe pins down how much of it is the arg surface),
plus whatever the gemm/gemv paths pay today; smaller per-dispatch
metadata for every tile-DSL kernel; one less custom launch abstraction
(`VecTensor`) to maintain.
