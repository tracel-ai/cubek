# Plan: resample as a client of the tile DSL

Goal: **retire the hand-rolled half of `cubek-resample` and unify `cubek-interpolate`.**
The crate manually decodes cube/tile coordinates from `FastDivmod` sequences,
hand-rolls its vectorization lanes, computes its own tile/cube geometry, and maps
output coordinates to input taps by hand. `cubek-tile` already owns every one of
those, and its gather path (`PhysicalAxisMap::affine`, `Compaction`, `AxisProjection`,
`Tile::nd`, gathered staging) was built for exactly this shape of operand.
Resample and interpolate become clients of it.

## Architectural Context

### The correction to the earlier plan

The first pass wrote resample as a **new tile op**, `Tile::resample`, with the
kernel weights folded into comptime `f32` constants. That was wrong: a resample
is `out[o] = ⊕_r w[r] ⊗ in[o·step + r·dilation − padding]`, which is a
gather-reduce over abstract window axes.

Folding weights into a comptime scalar deleted the second operand, breaking
multi-region accumulation, staging, leaf dispatch, and schedule integration.
Treating resample as an accumulation of gathered inputs against a weight operand
restores all standard tile mechanics (staging, double-buffering, scheduling).

### Weights as Procedural Tiles (no GMEM upload)

Rather than allocating and uploading filter weights (box, bilinear, bicubic,
lanczos, gaussian) as global memory (`Gmem`) tensors on every launch, weights
are evaluated on-chip via a **Procedural Tile** (`TileKind::Procedural`). The tile
computes weights analytically in registers from coordinates or comptime tables,
eliminating GMEM bandwidth and host upload overhead.

### What MMA can and cannot do (and why Reduce is needed)

- **Contraction resample (Add-Mul)**: Weights as `lhs`, output positions as
  batch axes, and input as gathered `rhs` runs directly on `Tile::mma`
  (`Leaf::Mma` or `Leaf::Cmma`) with no new leaf code.
- **Non-contraction semirings (`Tropical`, `Log`) and Pooling**: Max-pool,
  min-pool, and logsumexp are not contractions and have no MMA hardware
  support. They require a general **Gather-Reduce** op (`ops/reduce/`) in
  `cubek-tile`.
- **Separable resize**: Has a dense `(O×I)·(I×rest)` GEMM form against banded
  matrices. It always does more arithmetic than the gathered path ($O \cdot I$
  against $O \cdot R$, and the band width $R$ never exceeds $I$), so it only pays
  when MMA throughput outweighs that factor: heavy downsampling, where the kernel
  support grows with the scale factor and $R$ approaches $I$.

### Rational / Divisor Projections for Weird Scales

Arbitrary image scaling (continuous resize / interpolate with non-integer
scale factors, e.g. $100 \to 133$) requires rational coordinate mapping:
$\text{in} = \lfloor \frac{o \cdot W_{in} + \text{offset}}{W_{out}} \rfloor + r$.
Adding a **Divisor / Rational scale** to `PhysicalAxisMap` enables `Projection`
to express these fractional affine mappings directly.

---

## Completed Prerequisites (Done)

- [x] **Constant & Dynamic Offsets in `Projection`**: `Offset`, `PhysicalAxisMap::affine_with_offset`, threaded through `span`, `Compaction::of`, and `AxisProjection::to_source_pos`.
- [x] **Boundary Policies on Gathered Reads**: `Boundary::{Zero, Clamp}` supported on window reads.
- [x] **Launch-side Gather Entry & Dynamic Gathered Axes**: `StridedTileSource::gathered`, dynamic receptive-field dimension resolution, and fully dynamic launch support (`conv1d_launched_*`).
- [x] **Scalar Vector Relaxation**: Inner-most identity check relaxed for `vector_size == 1`.

---

## Steps

### Phase 1: Foundational Tile DSL Extensions (`cubek-tile`)

1. [x] **Rational / Divisor in `PhysicalAxisMap` & `Projection`**
   - Extend `PhysicalAxisMap` / `AxisTerm` with divisor / rational scale support (`(scale * x + offset) / divisor`).
   - Update `AxisProjection::to_source_pos` and `to_source_pos_checked` to execute integer division in-kernel.
   - Extend `Compaction`, `span`, and bounds checking / underflow analysis for rational projections.
   - Unlocks continuous resize / weird non-usize scales for `cubek-interpolate`.

2. **Procedural Tile (`TileKind::Procedural`) in `cubek-tile`**
   - Introduce a procedural backing store to `TileKind` and `Tile` that evaluates a comptime function / formula (or in-register table) given coordinate indices.
   - Satisfies operand traits so it can be passed directly as `lhs` to `Tile::mma` or as an operand to `Tile::reduce` without a backing `Tensor` parameter or GMEM allocation.

3. [x] **General Gather-Reduce Op (`ops/reduce/` in `cubek-tile`)**
   - Factor out the multi-axis gathered N-D register loop from `mma_register_gather` into a reusable reduction engine.
   - Implement `Tile::reduce` parameterized by semiring / combine-accumulate operators (`Add-Mul`, `Tropical Max/Min`, `LogSumExp`).
   - Supports `Gmem`, `Smem`, and `Procedural` tile inputs.

### Phase 2: Blueprint & Client Layer (`cubek-resample` & `cubek-interpolate`)

4. **Refactor `definition/` in `cubek-resample`**
   - Keep the existing public vocabulary (`Kernel`, `Semiring`, `BoundaryMode`, `NormalizationMode`), now describing the blueprint rather than driving an in-kernel tap loop.
   - Generalize `Placement` / `PlacementArgs` to cover rational and integer windowed geometry (`size`, `step`, `dilation`, `padding`, `scale`/`divisor`) in one description.
   - Delete the obsolete `CubeLaunch` arg wrappers (`ResampleArgs`, `ResampleAxisArgs`, `WindowArgs`, `PlacementArgs` launch types) and `accumulator.rs` (`Accumulator`, `Value`), whose job the tile accumulator takes over.

5. **`cubek-resample` Blueprint with Procedural Weights**
   - `blueprint.rs`: lowers resample specification and shapes to `Space`, input `Projection` (with rational `PhysicalAxisMap`), and `Procedural` weight tiles.
   - Eliminates host-side weight buffer allocations and tensor uploads.

6. **Unified Launch Surface in `cubek-resample`**
   - `launch/base.rs`: emits tile launches.
     - Contraction semiring (`Add-Mul`): dispatches `out.mma` or `out.reduce`.
     - Non-contraction semiring (`Tropical`, `Log`): dispatches `out.reduce`.
   - Delete the launch-side geometry helpers (`compute_tile_shape`, `compute_cube_shape`, `vectorize`, the `FastDivmod` sequences they feed).
   - Delete `components/` wholesale: `base.rs` (`resample_kernel`), `coordinates.rs` (the `FastDivmod` cube/tile coordinate decode), `tap_resolver.rs`, and `resample_instruction.rs`.

7. **Normalization Epilogue**
   - For static boundaries and `Boundary::Clamp`, fold normalizers directly into the procedural weight formula.
   - For `Boundary::Zero` with padding, emit an epilogue reduction pass to normalize by valid tap counts.

8. **`cubek-interpolate` as a Tile DSL Client**
   - Re-route nearest, bilinear, bicubic, and lanczos interpolation through the tile DSL using rational `Projection` and procedural weights.

### Phase 3: Optimizations

9. **Dense GEMM Form for Separable Resampling**
   - Add a second blueprint arm for separable 1D multi-pass resizing: $O \cdot I$ dense GEMM against banded matrices when downsampling factors make it faster than gathered MMA.

10. **Vectorization Fast Paths**
    - Enable vectorized paths when the innermost physical dimension is not contracted/resampled.

### Phase 4: Verification & Test Suite

11. **Comprehensive Test Suite**
    - Port existing 1D/2D resample and interpolate tests.
    - Test rational / weird-scale interpolation (e.g. non-power-of-2 scaling).
    - Test across all semirings (`Arithmetic`, `Tropical Max/Min`, `Log`).
    - Verify bit-exactness of procedural weights against reference tables.
    - Validate matrix of tile sizes, boundaries (`Zero` vs `Clamp`), paddings, and `Schedule`s (`Direct`, `Staged`, `DoubleBuffered`).

---

## Verification

- `cargo check` and `cargo clippy` on `cubek-tile`, `cubek-resample`, and `cubek-interpolate`.
- `cargo test -p cubek-resample`, `cargo test -p cubek-interpolate`, and `cargo test -p cubek-tile`, one package at a time. Never `cargo test --workspace`: it crashes the machine.

---

## Risks

- **Dynamic Rational Divisors**: Runtime integer division in index calculation has latency. Mitigate by keeping divisors comptime where possible or using fast division helpers when dynamic.
- **Innermost Axis Vectorization**: Scalar fallback is required when the innermost dimension is resampled. Verify perf on 1D resample workloads.
