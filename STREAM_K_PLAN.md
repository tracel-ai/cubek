# Stream-K in the tile DSL

Cutting the contraction so a cube does not do the whole of it. The `Tiling::over` half already
works; the whole feature is one missing concept on the drain, and that concept is one the DSL
already has at a finer scope.

## The claim, and what it rests on

`.axis(K, Cut::cube(CubeAxis::Z, k / splits))` compiles today and already multiplies the launch
grid (`space/partition/geometry.rs:37`). It also silently gives a wrong answer, two ways:

- `Residence::Register`: each cube accumulates its K slice in registers and `drain_cast_into`
  (`tile/plane.rs:359`) plain-stores over the same cells. Last cube wins, other partials lost.
- `InPlace`: `spans_contracted_at_leaf` is false, so `InitFrom::Cell`, so `AccumulateView::seed`
  reads the cell and `commit` writes it back. Lost-update race across cubes.

So nothing is missing from the tiling API. What is missing is the combine.

## The concept

`LaneShare` (`space/partition/distribution.rs:14`) already says the right sentence for `Unit`: an
axis my space does not span is folded across the instances, so each holds a partial, and the
drain is that scope's reduction. Swap the scope and the same sentence covers everything:

| scope of the omitted contracted axis | who holds a partial | drain | status |
|---|---|---|---|
| `Unit` | a lane | `plane_sum`, lane 0 writes | shipped |
| `Plane` | a plane | smem + `sync_cube`, plane 0 writes | missing, silently wrong |
| `Cube` | a cube | atomic add, or a workspace | missing, this is stream-K |

This is a unify-don't-add-vocabulary move: no new user-facing concept, the existing one grows a
scope. `Space::lane_share()` (`space/base.rs:496`) is already a mixed-radix digit question about
the instance index, filtered to `ComputeScope::Unit`. Filtering on a parameter instead is the
whole generalization.

## Split-K and stream-K are the same machinery

Flatten `(m_tiles x n_tiles x k_blocks)` with K innermost into a range of length `T`, give cube
`c` the run `[c*T/C, (c+1)*T/C)`. At `C = m_tiles*n_tiles*splits` every run lands inside one tile
and covers `k_blocks/splits` consecutive blocks: that is exactly split-K. So split-K is stream-K
whose runs never straddle a tile boundary, and the drain built for one is the drain the other
needs. Only the assignment differs: a product of per-axis runs versus a contiguous run of the
flattened grid.

Two consequences that shape the phases below:

1. The workspace design (bind the output over a real split axis, so no partial ever exists) is
   **rectangular-only**. It exploits every tile having the same fixed contributor count with a
   clean index. Stream-K has neither. The workspace trick does not survive the move; the atomic
   drain does.
2. Stream-K drags runtime-ness into a comptime design: run lengths vary, and tile ownership is a
   boundary comparison. Most of that is bought back by making the drain **unconditional** (every
   contribution goes through the combine, including wholly owned tiles). Then the only novelty
   left is the assignment.

## Phase 0: fail loud

Ship before any combine exists. Today three configurations are silently wrong: a contracted axis
at `Cube` scope, the same at `Plane` scope, and the known cmma-accumulator-plus-`Cut::unit`-on-K
gap where the combine cannot be seen from `CmmaData`.

- `Share` (`LaneShare` renamed scope-neutrally: `Whole` / `All` / `Group { fold_mask }`) and
  `Space::share(scope)`, the current body with the scope as a parameter.
- `MemData.lane_share` becomes one `Share` per scope, joined on `at()` exactly as now.
- `Tile::accumulate` panics when any scope's share is not `Whole` and no combine is declared on
  the operand, naming the scope and the axis.

Test: each of the three configurations panics with its own message. No behavior change otherwise;
`Unit` keeps its current path byte for byte, which the golden kernel-source diff proves.

## Phase 1: the split-K oracle, zero engine change

Cut K into `(KB, KI)` with `Map::disjoint` (`physical/projection/map.rs:210`), exactly as the
quant block axis does. Bind the output to a `[splits, M, N]` workspace that **spans KB**, with
`KB` on `Cut::cube(Z, 1)`. Now no operand omits a distributed axis, so there are no partials at
all: the drain is the plain vectorized store it is today. A second `Space` reduces over `KB` with
`reduce_axis`.

Needs nothing new. Its value is that it is the differential oracle every later phase is diffed
against, and the perf baseline the atomic drain has to beat.

Test: `split_k_workspace` in `tests/tile/matmul.rs`, arange reference, both the partial buffer and
the reduced result checked. Runs on CPU and Metal.

## Phase 2: the cube combine, through `Backing::WriteCall`

The seam already exists. `Backing::WriteCall(ErasedTensor<T, WriteOnly>)` (`tile/mem.rs:100`) is a
destination that is written through a call and never read, built for fused epilogues.
`ErasedTensorExpand::new<S: ErasedBacking<E, IO>>` is public and
`ErasedTensorOperationsExpand<E>::__expand_write_line_method` is the one method a backing must
serve. An atomic accumulate is precisely such a destination.

- `AtomicAccumulate<E, N>` in cubek-tile: a backing over an `Atomic<E>` binding whose `write_line`
  is `N` scalar `fetch_add`s. Implements `WritesLines<E>` and not `ReadsLines<E>`, so the type
  states that a partial is never read back. Entirely in cubek; no cubecl change.
- `AtomicTileArg<'a, E: Numeric, V: Size> { tensor: &'a Tensor<Atomic<E>>, spec }` with
  `.tile_accumulating(space) -> Tile<E>`, built on `Tile::of_sink`. Stored element and served
  element differ, which is the pattern `tile_packed` already establishes.
- The line width is unaffected at the tile level: the sink takes the stated width and decomposes
  inside the backing.

Gates, stated in the routine and refused rather than adapted to:

- `atomic_type_usage(Type::atomic(elem)).contains(AtomicUsage::Add)`. Verified present: metal
  native, wgpu MSL, CUDA sm60+, CPU, WGSL only under `SHADER_FLOAT32_ATOMIC`.
- accumulate type equals output type. f16 and bf16 atomic add exist only on CUDA, so an f16
  output takes phase 1's path.
- the output residence is `Register`. `WriteOnly` already makes `InPlace` a type error; this is
  the readable message rather than the confusing one.
- the launch owns zeroing the output, inside the routine, so `out = A*B` still holds at the
  routine boundary.
- reproducibility is a stated launch property, not a silent one. The strategy carries it and the
  atomic drain requires the value that permits reordering.

Test: same shapes as phase 1, diffed against phase 1's result within tolerance; a negative control
that forcing the cube share to `Whole` makes it fail. Metal first, since that is where the float
atomic needs proving.

## Phase 3: the plane combine

smem plus one `sync_cube`, plane 0 writes. Same `Share`, a different reduction. Independent of the
stream-K line and cheap, but it closes a real hole and it is the deterministic rehearsal for the
drain shape. Can land in parallel with phase 2.

## Phase 4: the linear assignment

- The assignment moves from the axis to the level. `LevelSpec` learns an `Assignment`, either the
  current per-axis one or a streamed one carrying the cube count; a streamed level's `LevelCuts`
  states edges only, since the distribution is no longer an axis's to state. Settle the surface
  before writing it: the alternative of a `Distribution::Streamed` per axis plus an all-or-none
  assert reads as a bool in a trench coat.
- `Walk::over_run(space, start, len)`: `from_counts` is already a mixed-radix decode, so this is a
  variant of the same body reading a linear cursor rather than a per-axis odometer.
- The drain stays unconditional. A fast path for wholly owned tiles is a measured optimization
  later, never a design requirement.
- The cube count is pinned with `Coverage::Instances(C)`, which already means what is needed, with
  `C` from `num_streaming_multiprocessors`.

Test: a run length that deliberately straddles tile boundaries, against the phase 1 oracle.

## Phase 5: selection and measurement

An eval category beside `tiled/eval/split_k/`, same shape, and the same rule: every mapping
verifies against a reference before it is timed, since a misconfigured mapping times fast and
means nothing.

Mappings: `DataParallel` (today), `SplitKWorkspace`, `SplitKAtomic { splits }`,
`StreamK { cubes }`. Heuristic to fit from the numbers: split only when
`m_tiles * n_tiles < waves * sms`, then `splits = clamp(target / tiles, 1, k_blocks)`. Expect to
keep both mappings and choose between them; they differ only in the `Cut`.

## What none of this touches

`mma`, the schedule, the leaf, the instruction, and the shape of `Tiling::over` for split-K. That
is how the lane-scope split landed and it is the test that this one is cut the same way.

## Risks

- Metal float atomic add at scale is the whole of phase 2. Probe it with a minimal kernel before
  building on it.
- The kernel cache hides comptime panics. Run the new configurations under `CUBECL_DEBUG_LOG` so
  they really compile.
- Non-determinism under atomics will move test tolerances. Decide the tolerance story in phase 2
  rather than discovering it in CI.
- Phase 1's workspace is `splits * m * n * 4` bytes. Bound `splits` or the path is unusable at
  large output.
- `of_sink` refuses staging, `dense_mut`, quantization and tensor-map fills. Confirm no planned
  output operand wants one of those before phase 2 commits to it.
