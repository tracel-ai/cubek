# Stream-K in the tile DSL

Cutting the contraction so a cube does not do the whole of it. The `Tiling::over` half already
worked; the feature was one missing concept on the drain, and that concept was one the DSL already
had at a finer scope.

Status: phases 0 to 3 landed and measured, plus the `mma`-not-`mm` correction that removes most of
what phase 4 looked like it would cost. Phase 4 itself (the linear assignment, which is what makes
it stream-K rather than split-K) is not started.

## What a caller writes

Split the contraction across cubes by cutting a contracted axis at cube scope, and drain into a
destination that folds:

```rust
let space = Tiling::new()
    .extents(&[(M, m), (N, n), (K, k)])
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
        l.axis(M, Cut::sequential(m))
            .axis(N, Cut::cube(CubeAxis::X, cols))
            .axis(K, Cut::cube(CubeAxis::Z, k / splits))   // the whole feature
    })
    .build()
    .with_instruction(Instruction::registers(64));
```

The kernel is the one an unsplit contraction writes. `AccumulateArg` is `TileArg`'s twin for an
output several instances add into, so only the argument's type differs:

```rust
fn matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    out: &AccumulateArg<'_, E>,          // the only difference
    #[comptime] space: Space,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = out.tile(space);
    c.mm(&a, &b, Semiring::SUM_PROD);    // still `mm`, still `c = a·b`
}
```

`mm` states `c = a·b` and owns the init that makes it true. Under a split a cell belongs to several
instances and none of them may seed it, so the init leaves the kernel: **the buffer arrives holding
the monoid's identity**, zeroed by the launch. Nothing can check that, since the destination cannot
read.

The deterministic alternative needs no atomics and no new engine: spell `K` as two axes and let the
output span the split, then fold it away in a second pass. `tests/tile/split_k.rs` has both.

## The concept

`LaneShare` already said the right sentence for `Unit`: an axis my space does not span is folded
across the instances, so each holds a partial, and the drain is that scope's reduction. The rest is
the same sentence at the scopes whose instances cannot meet in registers.

| scope of the omitted contracted axis | who holds a partial | drain | status |
|---|---|---|---|
| `Unit` | a lane | `plane_sum`, one lane writes | was already there |
| `Plane` | a plane | fold into the destination | landed |
| `Cube` | a cube | fold into the destination | landed |

`SplitShare` is `Plane` and `Cube` together, because they combine in the same place and by the same
means: each folds its own contribution into the destination and never learns that the others exist.
`Unit` stays with `LaneShare`, which needs a mask because lanes elect one of their own to write.

## Landed

**Phase 0, `e8cdc75c`.** `SplitShare`, `Space::split_share_of`, and the refusal. Three
configurations were silently wrong before it: a contracted axis at cube scope, the same at plane
scope, and the known cmma-plus-`Cut::unit`-on-K gap. A register-resident accumulator drains by
storing, so the last instance to arrive erased every other one's slice; one accumulating in place
read the cell, folded, and wrote it back, which is a lost update. Neither showed up as anything but
a wrong number.

**Phase 1, `66e90db2`.** Split-K spelled as an axis: `K` as `(KB, KI)` through
`PhysicalAxisMap::disjoint`, the output bound over `[KB, M, N]` so it spans the split, a second pass
folding `KB` away. Needed nothing new, and is the reference the in-kernel combine is diffed against.

**Phase 2, `bb8078a8`.** The atomic drain, through `Backing::WriteCall`. `AtomicAccumulate` is an
`ErasedTensor` backing whose `write_line` is `fetch_add`, so the walk, the layout, the masking and
the drain are the ones a plain store gets and only the last step differs. `Write` states what a
write means, since a backing cannot be asked: a folding sink and a fused epilogue are both calls
through a layout, and only the caller knows which it built.

**Phase 3, `0ed26312`.** The same at plane scope, which needed no new drain: the election is per
plane, so one lane of every plane of every cube folds its own contribution.

**Phase 5, `2c9b545f`.** `cargo bench -p benchmarks --bench split_cubes --features cubecl/metal`.

## What the numbers say

Medians, metal, two runs. Every mapping verifies against a reference before it is timed.

| shape | unsplit | workspace | atomic |
|---|---|---|---|
| m=1 n=32 k=8192 | 856us | 128us (/16) | **97us (/16)** |
| m=1 n=128 k=8192 | 465us | 186us (/4) | **164us (/4)** |
| m=1 n=512 k=8192 | 636us to 1.03ms | 456us to 1.09ms | 522us to 625us |
| m=8 n=128 k=4096 | 741us | 322us (/4) | **267us (/4)** |

The win is where it was predicted: too few output tiles to fill the device. `n = 32` at `COLS = 1`
is 32 cubes however deep `K` is, and splitting 16 ways is 8.8x. It fades as the output widens, and
`n = 512` is ambiguous. Atomic beats the workspace consistently but not hugely (1.1x to 1.3x), so
the second pass is cheap while the workspace is small; the case for atomics is that it is the drain
stream-K will also need, not that it is much faster here.

A split of one tracks the unsplit baseline, which is the control working. Wide shapes vary run to
run by up to 60%, the narrow rows repeat to the microsecond: compare within a run.

## What was learned that the plan did not predict

**A fold has to elect a writer even when nothing is folded across the lanes.** A space that rides no
`Unit` axis still launches a full plane, and its lanes all run the same work over the same cells.
Identical stores are idempotent, so nothing had to know; identical folds land `plane_size` times,
and the first atomic drain read exactly 32x. `LaneWork` is that fact, `Lanes` pairs it with the
share it is useless without, and `Drain` names the four elections once rather than nesting a lane
guard inside a share. The control is a case with `N` on the lanes and `K` on the cubes: a blanket
"lane zero writes" passes every other test and silently drops that one.

**A share cannot be derived from the operand's own projection.** The projection is exactly what
drops the contracted axis, so a projected space cannot tell a split from a cut whose edge happens to
be the whole axis, and the guard refused `Cut::cube(Z, k)` at `splits = 1`. Asked of the whole space
once, where the tile is built, and carried down unchanged the way `Write` is. Still conservative on a
`Dynamic` extent, where only the shape knows the tile count; refining that means stamping the share
host-side off the concrete space, the way bounds checks are already derived.

**Comptime panics land on a worker thread**, where `#[should_panic]` never sees them and the launch
returns zeros. Every refusal here is tested host-side instead, and `tests/tile/matmul.rs` carries the
note.

## Phase 4: the linear assignment

Split-K is stream-K whose runs never straddle a tile boundary, so the drain built above is the one
stream-K needs and none of it is throwaway.

**The accumulator-lifetime problem was overstated, and is now mostly gone.** The earlier reading
was that a cube's run crossing an output tile would force a mid-loop re-seed on a runtime check,
which meant `AccumulatorScope` giving up its lexical scope. What actually removes it is not
claiming ownership: an output that folds is *maybe prefilled*, every contraction into it is `mma`
rather than `mm`, and nothing has to prove a cube is the only contributor. `CellRead` answers
`Never` for an accumulating destination and the store's `fetch_add` is the read-modify-write, so a cube
adds its slice to whatever is there. Landed in `dc26fa6c`, with a test contracting in place into an
atomic output with no register accumulator at all.

A contiguous run over `(tile, k-block)` with `K` innermost is then a nest: a partial first tile,
whole middle tiles, a partial last tile. If the walk yields it that way, the accumulator stays
lexically scoped to the tile level and only the inner trip count becomes runtime. What remains:

- **The assignment stops factoring per axis.** `Coverage`/`Spread` are per-axis today and
  `Walk::from_counts` gives each cube a product of per-axis runs, which is a rectangular block. A
  contiguous run of the flattened grid is not one. `Coverage` has to move from the axis to the
  level, and `Walk` needs to decode from a linear cursor rather than an odometer, which is a
  variant of the mixed-radix decode it already does. Settle the surface first: a
  `Distribution::Streamed` per axis plus an all-or-none assert reads as a bool in a trench coat.
- **A runtime trip count on the inner loop**, which the walk already half-supports (`steps` folds
  to a constant only when the grid is comptime).

## The lane question, which is Louis's and is not settled

A cube launches a full plane whatever the space says. A space that rides no `Unit` axis therefore
runs `plane_size` copies of one lane's work and throws all but one away, and `split_k`'s `seq_k`
mapping already knows: it overrides `CubeDim::new_single()` rather than waste them.

That is a pre-existing waste, but the fold made it a *correctness* question, twice: repeated lanes
storing the same value land it once, and folding it land it once each. Both write paths needed a
lane-zero election (`Drain::LaneZero`), which is silent adaptation to a misconfigured space, and
the house rule is the opposite: refuse, and let the caller put something on the lanes.

Measured, it matters more than the split does. On `m=1 n=32 k=8192`, cutting each cube's slice of
`K` again across the plane (`atomic_lanes`) is **53us**, against 97us for the best mapping that
leaves the lanes idle and 475us unsplit. Most of what phase 5 first read as the cube split's win
was the lanes waking up.

What blocks turning the election into a refusal: `LaneWork` is derived from the space at comptime
and cannot see `plane_size`, so `Repeated` is genuinely correct on the CPU runtime (one lane) and
wasteful on a GPU (32). A refusal therefore belongs host-side, at launch, where `Space::cube_dim`
already asks the client for `plane_size` and already asserts about `Unit` counts. It would need to
know the destination folds, which is the operand's statement rather than the space's. Louis's call.

## Still open

- The lane-zero election above is silent adaptation, and probably wants to be a launch-side
  refusal instead.
- Reproducibility is not yet a stated launch property. The atomic drain reorders, so its result is
  not bit-identical run to run, and nothing says so at the call site.
- `f16`/`bf16` outputs take the workspace path: their atomic add exists only on CUDA. Unenforced;
  binding a non-`AtomicUsage::Add` element will fail in the backend rather than at the launch.
- The share stays conservative on a `Dynamic` extent, so a launcher-built space always folds. Fix is
  host-side stamping off the concrete space.
- `an_i8_operand_contracts_against_its_scales` fails on metal, and did before any of this. Unrelated
  (quantized i8 against scales), untouched.
