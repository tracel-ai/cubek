# Splitting cubek-matmul into `tiled` and `multi_level`

## Goal

`cubek-matmul` gets one branch point at its root. Everything below it is either
tile-DSL work (`tiled/`) or level-tower work (`multi_level/`). Each side compiles
without the other, and every test lives under one of the two.

The tiled side becomes first class: no component tower, no `args` trait zoo, no
fake tiling borrowed from `cubek-std`. The other side keeps working (several of
those kernels are still the fastest we have) but sits in its own island, so that
when the tile DSL wins everywhere the island can be deleted in one move.

Never write "legacy" in the code. The separation carries the meaning; the words
do not.

## Where the seam already is (measured)

The good news: the tiled routines barely touch the tower today.

`routines/cmma/` and `routines/cpu_gemm/` (8 files) import from `crate`:

- `definition::{MatmulProblem, MatmulElems, MatmulSetupError, MatmulAvailabilityError, AvailableVectorSizes, broadcast_batches}`
- `routines::{Routine, BlueprintStrategy, DeviceSettings, M, N, K, batch_axis, MatmulOperands}`
- `components::global::read::stride_align_bits` — the single genuine leak, one pure
  8-line function about TMA stride alignment.

Everything else in `cubek-matmul` is tower-side: `components/` (12.8k lines, 121 files
with the multi-level routines), `args.rs` minus its `RuntimeConfig` blanket trait,
`routines/{batch,gemm,gemv_unit_perpendicular,naive,selector}`, `strategy/select_kernel.rs`,
and about two thirds of `definition/`.

`cubek-std` splits the same way: `tile/` (5.1k lines) plus `stage/` (0.9k) are the
matmul fake tiling; `cube_count/`, `input_binding.rs`, `matrix_layout.rs`, `error.rs`,
`launch/tma.rs`, `layout/`, `stage_ident.rs` are host-side utility that both branches
and `cubek-reduce` need.

## Decisions

| Question | Decision |
|---|---|
| Root branch shape | Nested enum: `Strategy::Tiled(..)` / `Strategy::MultiLevel(..)`, each `#[cfg]`'d |
| `gemm`, `gemv_unit_perpendicular`, `naive` | Into `multi_level/` — they need `components::batch` + `args`, so they die with it |
| `cubek-std` | Shrink and keep. Matmul-specific parts move back into `cubek-matmul/src/multi_level/` |
| attention / convolution | Re-point imports at `cubek_matmul::multi_level::*`, declare `features = ["multi-level"]`. No internal branch this round |

## Target shape

```
crates/cubek-matmul/src/
  lib.rs                    # the branch: two cfg'd modules + the shared root
  launch.rs                 # launch_ref(&Strategy, ..)
  strategy.rs               # the nested Strategy enum, Display, dispatch
  tune_key.rs               # MatmulAutotuneKey (branch-agnostic)
  routine.rs                # Routine, BlueprintStrategy, DeviceSettings,
                            # RuntimeConfig, M/N/K, batch_axis, MatmulOperands,
                            # into_contiguous_if_highly_permuted, num_concurrent_planes
  definition/               # MatmulProblem, MatmulKind, MatmulIdent, MatmulElems,
                            # MatmulGlobalElems, errors, vectorization, cost,
                            # broadcast_batches
  eval/cpu_reference.rs     # branch-agnostic reference

  tiled/                    # #[cfg(feature = "tiled")]
    mod.rs
    strategy.rs             # enum { Cmma(..), CpuGemm(..) }
    cmma/
    cpu_gemm/
    eval/                   # gemm_cpu_tiled, tile_quant_stage catalogues

  multi_level/              # #[cfg(feature = "multi-level")]
    mod.rs
    strategy.rs             # the ~40-variant enum, moved verbatim
    args.rs                 # MatmulArgs, TensorArgs, TensorMapArgs, Concrete*Factory
    select_kernel.rs
    definition/             # BatchMatmulBlueprint, TilingScheme, CubeMapping,
                            # MatmulPrecision/MatmulTypes + the type-alias zoo
    routine.rs              # BatchMatmulRoutine, ExpandInfo, LaunchInfo,
                            # batch_validate_blueprint, TilingArgs
    components/             # batch / global / stage / tile, unchanged
    tile/                   # from cubek-std/src/tile
    stage/                  # from cubek-std/src/stage
    plane_flow.rs instruction.rs cube_dim_resource.rs size.rs   # from cubek-std
    routines/{batch,gemm,gemv_unit_perpendicular,naive,selector}
    eval/                   # gemm, gemv, split_k, quantized_matmul catalogues
```

```rust
// lib.rs
#[cfg(feature = "tiled")]
pub mod tiled;
#[cfg(feature = "multi-level")]
pub mod multi_level;
```

```rust
// strategy.rs
pub enum Strategy {
    #[cfg(feature = "tiled")]
    Tiled(tiled::Strategy),
    #[cfg(feature = "multi-level")]
    MultiLevel(multi_level::Strategy),
    #[default]
    Auto,
}
```

`cubek-std` after the move:

```
crates/cubek-std/src/
  cube_count/     input_binding.rs   matrix_layout.rs
  error.rs        launch/tma.rs      layout/          stage_ident.rs
  eval/           # contiguous, unary, memcpy_async benches (already independent)
```

`MatmulProblemSize` stays (the autotune key needs it). `TileSize` / `PartitionSize` /
`StageSize` move to `multi_level/size.rs`; the `define_3d_size_base!` macro stays in
`cubek-std` and gets `#[macro_export]`ed so both sides can use it.

## Phases

Each phase is one landable change that compiles green with both features on. Only
phase 6 flips CI to enforce the seam.

### Phase 1 — cut the leaks, no moves  ✅ DONE

Pure file splits inside the current tree. Nothing changes location yet, so review is
about the split lines, not about `git mv` noise.

1. Move `stride_align_bits` from `components/global/read/strategy/base.rs` to
   `cubek_std::launch::tma`. Kills the only tiled → tower import.
2. Split `routines/base.rs` (213 lines):
   - stays shared: `Routine`, `M`/`N`/`K`, `batch_axis`, `MatmulOperands`,
     `into_contiguous_if_highly_permuted`, `DeviceSettings`, `num_concurrent_planes`
   - tower-only: `BatchMatmulRoutine`, `ExpandInfo`, `LaunchInfo`,
     `batch_validate_blueprint`
3. Split `args.rs` (550 lines): `RuntimeConfig` (2 lines, blanket impl) stays shared;
   `MatmulArgs`, `TensorArgs`, `TensorMapArgs`, `ConcreteInputsFactory`,
   `ConcreteOutputFactory`, `TensorInputIdent` go tower-side.
4. Split `definition/`:
   - shared: `base.rs`, `error.rs`, `vectorization.rs`, `cost.rs`, and
     `MatmulElems` / `MatmulGlobalElems` carved out of `spec.rs`
   - tower: `blueprint.rs`, `tiling_scheme.rs`, `cube_mapping.rs`, and the rest of
     `spec.rs` (`MatmulPrecision`, `MatmulTypes`, `Lhs`/`Rhs`/`Acc` alias zoo)
5. Split `routines/selector/`: `BlueprintStrategy` shared; `TilingArgs`, `plane.rs`,
   `unit.rs` tower-side.
6. Split `strategy/`: `tune_key.rs` shared; `select_kernel.rs` and `test_only.rs`
   tower-side; `strategy.rs` into two enums plus the root wrapper.

Watch item: `MatmulIdent::into_stage` returns `StageIdent`. Keeping `stage_ident.rs`
in `cubek-std` (26 lines) avoids a shared → tower dependency here.

**What actually landed**, and where it deviated from the sketch above:

- `RuntimeConfig` now lives in `routines/base.rs`; `args.rs` re-exports it. A tower module
  re-exporting a shared item is the allowed direction, and it kept ~40 tower files untouched.
- `MatmulElems` / `MatmulGlobalElems` moved to a new `definition/elems.rs`. Two methods stayed
  tower-side as second inherent impls in `definition/spec.rs`: `MatmulElems::new_deprecated`
  (needs the precision traits) and `MatmulIdent::view_direction` (returns
  `components::global::memory::ViewDirection`). That second one was an unlisted shared → tower
  leak; it is now gone from `definition/base.rs`.
- `BlueprintStrategy` moved out of `routines/selector/base.rs` into `routines/base.rs`, leaving
  `selector/base.rs` holding only `TilingArgs`.
- `strategy/` gained `tiled.rs` and `multi_level.rs`; `strategy/strategy.rs` is now the nested
  root enum plus `From<tiled::Strategy>` / `From<multi_level::Strategy>` and `auto`.
- Call-site churn from the nested enum: `Strategy::X(..)` became
  `MultiLevel::X(..).into()` across 21 test files and 3 bench catalogues. `Strategy::Auto` and
  the harness signatures were untouched.
- Found dead on the way through: `definition/cost.rs` (`MatmulCost`, `Work`, `compute_key`) has
  no callers anywhere in the workspace. Left in place; worth deleting separately.

Seam verified by grep, both directions: `routines/cmma`, `routines/cpu_gemm`, and
`strategy/tiled.rs` contain no reference to `components::`, `args::`, `BatchMatmul*`,
`TilingScheme`, `MatmulTypes`, `MatmulPrecision`, `cubek_std::tile`, or `cubek_std::stage`.
Same for the shared root (`routines/base.rs`, the five shared `definition/` files,
`strategy/strategy.rs`, `strategy/tune_key.rs`, `launch.rs`). And the reverse: the tower
names no `cubek_tile` item and `strategy/multi_level.rs` names no tiled routine.

Verification for this and every later phase: `cargo check` / `cargo clippy`, including
`--features extended,benchmarks --tests` which *compiles* the extended tier. Never *run* the
`extended` or `full` test tiers, they take far too long. `full` additionally OOMs rustc on a
16 GB machine, pre-existing and reproduced identically on a pristine `HEAD` worktree.

### Phase 2 — introduce the two modules  ✅ DONE

`git mv` only, plus import fixups. `tiled/` gets `cmma/`, `cpu_gemm/`, its strategy
enum. `multi_level/` gets `components/`, the four tower routine families, the selector,
`args.rs`, its half of `definition/`, `select_kernel.rs`, its strategy enum. Root keeps
`launch.rs`, `strategy.rs`, `routine.rs`, `definition/`, `tune_key.rs`.

`launch_ref` becomes a 3-arm match. No `#[cfg]` yet: both modules always compile.

**What actually landed.** 139 renames, 265 files touched. The crate root is now five files
(`lib.rs`, `launch.rs`, `routine.rs`, `strategy.rs`, `tune_key.rs`) plus `definition/`, `eval/`,
and the two branch directories. `src/routines/` and `src/strategy/` are gone.

Path moves beyond the sketch:
- `routines/base.rs` → `routine.rs` at the root; `routines/batch_base.rs` → `multi_level/routine.rs`.
- `strategy/tune_key.rs` → `tune_key.rs` at the root, so `MatmulAutotuneKey` is no longer reached
  through `strategy`. No in-workspace consumer; burn's import path changes.
- `strategy/test_only.rs` → `multi_level/test_only.rs`.
- `definition/{blueprint,cube_mapping,spec,tiling_scheme}.rs` → `multi_level/definition/`.
  `CubeMapping` / `CubeMappingLaunch` / `cube_mapping_launch` are re-exported from there.

Public path changes for downstream (conv and attention were ported in this phase; burn is not):
`cubek_matmul::components` → `::multi_level::components`, `::args` → `::multi_level::args`,
`::routines::<family>` → `::multi_level::routines::<family>`, `::routines::{Routine,
BlueprintStrategy, DeviceSettings}` → `::routine::{…}`, the tower half of `::definition` →
`::multi_level::definition`, `::strategy::{launch_kernel*}` → `::multi_level::{…}`,
`::strategy::test_only` → `::multi_level::test_only`.

Two spots needed hand-work rather than a path rewrite: `routines/batch/{double_buffering,
specialized}.rs` referred to `base::Routine` / `batch_base::BatchMatmulRoutine` as *module*
aliases inside macro bodies (now the traits are named directly), and six tower files import
`crate::definition::*` as a glob, which now needs `crate::multi_level::definition::*` alongside it.

Verified: `cargo check` on the whole workspace `--all-features`; `--tests` on every crate except
matmul's `full` tier; `--benches` on the benchmarks crate; `clippy` clean on matmul (extended +
benchmarks + tests) and on convolution / attention / std / tile; `cargo fmt` clean.

Seam re-verified after the move, three directions: `tiled/` names nothing from the tower,
the shared root names neither branch, and `multi_level/` names neither `crate::tiled` nor
`cubek_tile`. Exactly three files name both branches: `lib.rs`, `strategy.rs` (the branch point),
and `eval/benchmarks/gemm/strategy.rs` (Phase 7 splits it).

### Phase 3 — demolish cubek-std's matmul half

Move into `cubek-matmul/src/multi_level/`: `tile/` (44 files with `stage/`), `stage/`,
`plane_flow.rs`, `instruction.rs`, `cube_dim_resource.rs`, and the three tiling size
types out of `size.rs`.

Re-point:

- `cubek-attention`: `cubek_std::{tile, stage}` → `cubek_matmul::multi_level::{tile, stage}`;
  `cubek_matmul::components::*` → `cubek_matmul::multi_level::components::*`.
  Add `cubek-matmul = { features = ["multi-level"] }`.
- `cubek-convolution`: same, plus `cubek_matmul::args` → `multi_level::args`,
  `cubek_matmul::strategy::launch_kernel` → `multi_level::launch_kernel`,
  `cubek_matmul::routines::*` → `multi_level::routines::*`.
- `cubek-reduce` keeps its `cubek-std` dep (uses `cube_count` only).

`cubek-attention` and `cubek-convolution` still depend on `cubek-std` for `cube_count`
and `launch::tma`.

### Phase 4 — feature flags

`cubek-matmul`:

```toml
[features]
default = ["std", "tiled", "multi-level", "cubecl/default"]
tiled = []
multi-level = []
```

`#[cfg]` the two modules, the two `Strategy` variants, their match arms, and the
per-branch `eval/benchmarks` catalogues. `cubek-attention` / `cubek-convolution`
declare `features = ["multi-level"]` explicitly rather than relying on `default`.

`cubek` facade gains `matmul-tiled` / `matmul-multi-level` passthroughs.

`Auto` needs a decision per compiled set:

| compiled | `Auto` |
|---|---|
| both | unchanged: SimpleCyclicCmma, falling back to SimpleUnit |
| multi-level only | unchanged |
| tiled only | Cmma, falling back to CpuGemm |

`CpuGemm` is the only tiled routine with no hardware requirement, so it takes the
fallback slot. It was tuned for CPU and has never been measured as a GPU fallback;
we wire it now and measure later. Until then the tiled-only `Auto` path is correct
but of unknown speed on accelerator-less GPUs.

### Phase 5 — tests

```
crates/cubek-matmul/tests/
  lib.rs
  harness/          # run(), run_with_strides, assert_result, problem builders
  tiled/            # #[cfg(feature = "tiled")]
    basic/ extended/ full/
  multi_level/      # #[cfg(feature = "multi-level")]
    basic/ extended/ full/
```

Rule: zero `#[test]` functions outside `tiled/` and `multi_level/`. `harness/` holds
launcher plumbing only; it is branch-agnostic because `run` takes a closure. It is the
one shared thing under `tests/`, and it carries no test of its own.

Each branch gets a thin `test_strategy(client, problem, <branch>::Strategy)` wrapper so
tests do not spell `Strategy::Tiled(tiled::Strategy::Cmma(..))` at every call site.

Assignment of the 64 existing test files:

- tiled: `extended/cpu_gemm/*`, the `Strategy::Cmma` cases currently mixed into
  `basic/plane_accelerated.rs`, the tiled half of `bench_catalog.rs`
- multi_level: `basic/{auto,gemv,naive,plane_vecmat,tma,unit}`, the rest of
  `basic/plane_accelerated.rs`, `extended/{advanced,alt_shapes,layouts,quantization,
  stride_zero,tiling_scheme,gemm}`, all of `full/`, `bias.rs`
- harness: `launcher_strategy.rs`, `basic/common.rs`, `extended/common.rs`

`basic/plane_accelerated.rs` is the only file that must actually be cut in two.

The `heavy` / `extended` / `full` tiers stay as they are and cross with the branch
axis, not replace it.

### Phase 6 — enforcement

Without a CI job the cfg attributes rot within a week. Add to the matrix:

```
--no-default-features --features tiled,std,cubecl/cpu
--no-default-features --features multi-level,std,cubecl/cpu
--features cubecl/cpu                          # both, existing
```

for `build`, `test`, and `clippy`. The tiled-only job is the one that proves the goal;
the multi-level-only job proves we did not accidentally make the tower depend on the
tile DSL.

`xtask test` learns the same three combos for `cubek-matmul`.

### Phase 7 — benchmarks and eval

`eval/cpu_reference.rs` stays at the root (branch-agnostic). Catalogues split:
`gemm_cpu_tiled`, `tile_quant_stage` → `tiled/eval/`; `gemm`, `gemv`, `split_k`,
`quantized_matmul` → `multi_level/eval/`. `gemm_cpu` names strategies from both, so it
either splits or gets per-entry cfgs.

The `benchmarks` crate keeps both features on.

## Open items

1. **Tiled-only `Auto` is unmeasured.** Decided: wire `Cmma` → `CpuGemm` now, measure
   later. Owed work, not a blocker: bench the tiled-only `Auto` path on wgpu/metal
   against `SimpleUnit` once the split is green, and revisit if it is bad enough to
   prefer a loud `Unavailable`.

2. **burn takes the break, with `multi-level` on.** Every `Strategy::SimpleUnit(..)`
   call site becomes `Strategy::MultiLevel(multi_level::Strategy::SimpleUnit(..))`. No
   compatibility shims: burn declares `cubek-matmul` with `features = ["multi-level"]`
   and stays on the tower for as long as it needs. A shim layer would re-blur the branch
   and someone would have to delete it later.

3. **`RuntimeConfig` on the tiled side.** Deferred. Every tiled use is
   `BlueprintStrategy<(), X>`; the `RC` parameter exists only for the fusion path.
   Dropping it from `Routine` / `BlueprintStrategy` on the tiled side is cleaner but is
   a real API change, and phase 1 is otherwise pure file-splitting. The blanket trait
   stays in the shared root; revisit once the split is green.

4. **`bias.rs` is the only test reaching into `components::global::memory`.** It builds
   launch args by hand. It moves to `multi_level/`, but it is a sign that the tower's
   test surface leaks into internals; not this refactor's problem.

5. **attention and convolution eventually need the same branch.** Their tiled sides
   exist today only as prototypes in `cubek-tile/tests/tile/{attention,conv}.rs`. This
   plan leaves them as pure multi-level clients, which is honest and keeps them in the
   deletable island.

## What this buys

When the tile DSL wins everywhere, deletion is:

```
rm -r crates/cubek-matmul/src/multi_level
rm -r crates/cubek-matmul/tests/multi_level
```

plus dropping the feature, the `Strategy::MultiLevel` variant, and whatever
`cubek-attention` / `cubek-convolution` have not yet ported. Nothing in `tiled/` or at
the root has to be touched.
