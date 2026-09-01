# Quantization in the tile API: the explicit route

Where a quantized operand's pieces are named, and what is left to build.

## The rule

**Scales are an operand. Folding them in is a verb.** A quantized tensor is values *and* scales;
an operand's binding can name one thing, so the other cannot ride it. `QuantTileArg`,
`Quantization`, `DequantAt` and `QuantInfo` are the machinery of smuggling the second one through
the first, and they go. What replaces them is a kernel that says what it reads:

```rust
let w = w.tile(space);            // Tile<u32>, honestly words
let s = s.tile(space);            // Tile<f16>, honestly scales
c.mm_scaled(&w, &x, &s, Semiring::SUM_PROD);
```

Everything follows from that. A scale's element type is the type of the tensor bound, because it
*is* a tensor — which is the whole f16-scales problem, dissolved rather than solved. The decode
site is where the verb is written, so `validate_dequant_at`'s table of legal pairings has nothing
left to police: an illegal pairing has no spelling.

The design record is the 2026-08-20 session's, "Space or Verb":
<https://claude.ai/code/artifact/734748ee-f396-4eb0-9410-04793f17b611>. The rule above is that
record's one line — *the space may state structure and placement; anything needing an operand the
space cannot name has to be a verb in the kernel* — pointed at quantization.

## Landed

1. **The coarse operand needs no engine work** (`tests/tile/coarse.rs`). A scale resolves at a
   coarser level of the same axes, and the plan called that the keystone, to be built as new
   digit-decomposition work. It already exists: a rational projection *is* it.
   `PhysicalAxisMap::of(K).over(block)` is `⌊k / block⌋`, the same floor the resample mapping
   rides. Proven with plain floats at three cuts — equal to the block, finer (several regions read
   one value), coarser (one region addresses several).

   `Tile::copy` refuses a coarse source, and that is the right refusal: a compacted stage fill
   requires source and destination to share a projection. A scale is consumed where the values
   are, never staged into the shape of its own expansion.

2. **`c.mm_scaled(&a, &b, &s, semiring)`** (`ops/matmul/lower.rs`, `tests/tile/scaled.rs`).
   `c = (a ⊗ s) · b`, the scale entering on the product side — one more factor of the term, not
   one more term. The scales ride the walk *beside* the ring rather than in it, so nothing needs a
   three-operand `Staging` payload (the standing open question): one value per block is
   cache-served wherever it sits, and staging it would materialize the expansion the coarse read
   avoids. f16 scales are one of the tests.

   Duplicated bodies rather than an option on the plain path — `contract_scaled` /
   `rank1_update_scaled` (`instruction/registers/block.rs`), `contract_scaled` / `nest_scaled` /
   `body_scaled` (`contract/direct.rs`), `memory_scaled` (`contract/base.rs`). The plain
   contraction is the hot loop of every matmul in the crate, and a scaled step is a different
   program: one more read and one more product per value.

3. **`TileSpec::packed(field)`: packed values without a scheme** (`tests/tile/packed.rs`).
   `Packing` is self-describing now — `Packed { field: QuantValue }`, the field naming its own
   width and sign — and it is *stated* on the operand rather than recovered from a scheme.
   `w.tile_packed::<E>(space)` serves a `u32` binding's fields as values, unpacked at the read by
   `PackedView` (`tile/view/packed.rs`), the unscaled twin of cubecl's `QuantizedView`: no scheme,
   no scale binding, no block grid anywhere in the path.

   `Store` carries the packing, so it is one statement whichever door minted it: a quantized
   operand's scheme derives it (`scheme_packing`), a spec states it, and every "stored is not
   served" refusal keys on it rather than on the presence of scales. A packed source also unpacks
   under `copy_from`, which is decode-into-smem for free when it stages.

   The q4 kernel the plan was blocked on now runs end to end:
   `w.tile_packed()` + a scales tensor + `c.mm_scaled(&w, &x, &s, ..)`. Q4S/Q2S need a device whose
   vectors reach the packing factor (a packed line serves a whole word); verified on cpu, Q8S runs
   everywhere.

4. **The side is read, not stated** (`ScaleSide`, `tests/tile/scaled.rs`, `tests/tile/packed.rs`).
   `mm_scaled` scales whichever operand the scales' *own axes* name: one spanning the output's
   columns is a fact about the rhs's columns and nothing else could fold it in; anything else
   scales the lhs. Same verb, same kernel body, both sides — `(a ⊗ s) · b` and `a · (b ⊗ s)` are
   the same sum of terms, and the side is only where the factor folds in cheapest (once per
   `(row, k)`, or once per `(col, k)` into each rhs line).

   `n` counts the rhs's lines while the scales count their own values, so the rhs read widens the
   column by the accumulator's width; a line still may not straddle a block. A scale over *both*
   matrix axes is refused: that is a scale of the output, not a factor of either term.

5. **The promoted accumulator, and the line rule enforced** (`tests/tile/scaled.rs`).
   `mm_scaled` reaches a register-resident accumulator: `AccumulatorScope::mm_scaled` /
   `mma_scaled`, `PlaneTile::mma_scaled`, `RegisterData::mma_scaled`. The scaled partials no
   longer round-trip through the sink's element between `K` steps, which is the form the decode
   gemv wants. A fragment accumulator still refuses, for the reason under Deferred.

   And the rule the docs stated is now checked: `check_lines_hold_one_scale` reads the block off
   the scales' own projection (`scale_block`) and refuses a step whose line straddles two. Which
   axis the line runs along is the step's own business — `K` past one served value, the
   accumulator's columns at one.

## The block is an axis

An axis exists when an operand *distinguishes* it. A quantized operand's scales vary over the block
index and not over the position inside the block, so those are two axes:

```
axes:   M, N, KB, KI          extent(KB) = K/B,  extent(KI) = B

A (values)   (M, KB, KI)   physical [M, K],   K = digits(KB, KI)
S (scales)   (M, KB)       physical [M, K/B], direct
B (rhs)      (KB, KI, N)   physical [K, N],   K = digits(KB, KI)
C (out)      (M, N)        direct
```

The scales stop being special: they are the operand that omits `KI`. Invariance is then
*structural* (omission), not derived — no divisor, no `⌊k/B⌋`, no `ScaleLayout`, and no rule about
what a line may straddle, because a cut cuts `KB` or cuts `KI` and cannot straddle two axes.
Depth of the scale hierarchy = number of splits, so two levels is `digits(KBO, KBI, KI)` and
recursion does the rest.

You split only the axes with a *non-trivial* block: per-tensor and per-row need none (they are pure
omission), the common quant cases need one, a 2-D block needs two. A split propagates to every
operand spanning that axis, unquantized ones included — that is the rank of the problem, not
overhead.

**Landed (phases 0 and 1).** `Composition` names whether a physical axis's terms *tile* it or may
overlap on it; `digits` claims the first, `affine` the second, and the identity is derived rather
than claimed. `Projection::validate_composition` checks the claim against the extents where the
projection and the space first meet. Three gates then stop calling a partition a gather:
`Tile::gathered` asks the composition; the matrix view takes the groups it reads over
(`MatrixGroups`) instead of assuming the last two axes, which collapsed `BatchMatrix` and
`GroupedMatrix` into one layout over a batch prefix, a row group and a column group; and the leaf
routes on whether those groups *exist* rather than on an axis count. `tests/tile/blocked.rs` pins
it with no quantization in sight.

Two things fell out: split-K and the two-contracted-axis matmul now take the *direct* nest, because
their contracted axes do form one `k` edge; and the block-vs-cut alignment rule is now enforced by
construction rather than by an assert.

**Landed (phase 2).** `tests/tile/scaled.rs` and `tests/tile/packed.rs` are spelled by omission:
eleven specs that carried a divisor carry none. Four places learned that a spanned but *unmapped*
axis is a broadcast rather than a malformed projection — `validate` refuses an axis addressing
several physical axes and no longer one addressing none, `tiling` reads it as untiled,
`bound_states` reports no dim to read a bound off, and `Projection::addresses` names the state.
`validate` also accepts a partitioned innermost axis under a vector line, and `strided_2d` asks
for a `k` edge instead of counting axes.

The scaled nest was never given `ContractShape`, so it had been deriving `kc` from the first
contracted axis rather than the product and routing on an axis count. Both are the plain path's
answers now. That drift is what a duplicated program costs.

**What still blocks the deletions.** `ContractShape` derives the *accumulator's* rows and columns
from its last two axes, so splitting an axis the output spans (`N`, for a `[bm, bn]` scheme) sizes
the block wrong. Nothing states the output's edges the way `MatrixAxes` states the operands'. Until
that exists, a scheme that blocks the columns keeps the rational spelling there, and
`check_lines_hold_one_scale` plus `ScaleSide` keep earning their place.

## Next, in order

| # | item | notes |
|---|---|---|
| 1 | **the scale rides the operand** | `c.mm(&a.scaled(&s), &b, semiring)`. `mm` goes generic over an `Operand` trait instead of taking `&Tile` outright, and `Scaled<T, S>` is a pair of tiles implementing it. That deletes twelve `*_scaled` functions (`mm`/`mma`/`mma_leaf`/`PlaneTile::mma`/`mma_buffered`/`AccumulatorScope::{mm,mma}`/`memory`/`contract`/`nest`/`RegisterData::mma`/`matrix`) whose only job is carrying a third argument down eleven frames to one call site. It also deletes `ScaleSide` and `scale_side`: the caller states the side by writing `.scaled()` on the operand that has the scales, instead of the engine inferring it from axes. **Proven** (`tests/spike_operand.rs`, delete when the spine lands): a `#[cube]` trait carries both a `comptime_type!` return and a `-> Self` return, and the unscaled path through the trait compiles to byte-identical kernel source, SSA numbering and unrolled loop included. Two footguns: `Scaled` must own its tiles, not borrow them, because `Tile::at` returns fresh tiles; and `Scaled::<T, T>` does not parse (`#[cube]` reads a repeated turbofish param as a redeclaration), so the two element types must differ, which is the real case anyway |
| 1b | **scales are vectorized, and the lane is the leaf's** | A scales line of `SW` lanes covers `SW` blocks: `SW · B` values, `SW · B / V` value lines, with value line `j` taking lane `j · V / B`. That division is exact because a line already may not straddle a block. **Decided: the leaf holds the lane, not the view.** `ScaledView` would have to derive the lane from a read position, which is only free if the position folds to a constant; `block::contract`'s `lane_fanout` walk already loads a line once and walks its lanes under a comptime index, which is exactly the shape wanted. So `ScaledView` keeps the `SW == 1` broadcast and the wide case belongs to `block::contract`. Consequence for item 1: `Operand`'s method is *run*-shaped (hand the leaf a run of lines), not view-shaped, so the spike's `matrix_of` is the wrong granularity. Today scalar scales are nailed shut in two places, and the assert in `direct.rs` guards that rather than stating anything: `direct.rs` binds the scales' width to `1`, and `scale_line` does `extract(0)`, using lane 0 and discarding the rest |
| 2 | **the accumulator's edges, by membership** | the head of the queue, and what vectorized scales wait on. See below |
| 3 | **the scales on the ring** | a split scales operand stages one value per block, no expansion, so it can ride the staging ring like any operand and `MmaScaledWalk` merges back into `MmaWalk`. Needs a ternary `Ring`/`Staging`; the verb itself stays, since a store cannot hold a store (`Box<MemData<T>>` is not a `CubeType`) and the alternative forks `MemData::at` |
| 4 | **two levels** | `disjoint(&[(KBO, ..), (KBI, ..), (KI, 1)])` for mxfp4's shape: one more label in the same list, since mixed radix does not care how many digits |
| 5 | **port the quant tests, then delete** | `QuantTileArg`, `Quantization`, `DequantAt`, `validate_dequant_at`, `QuantInfo`'s block bookkeeping, `flat()`'s dequantizing read, `copy_from`'s arithmetic. Acceptance: identical numbers on every existing quant test. **Not a mechanical port** — see the survey below |
| 6 | **the N-D nest, if anything asks** | a genuinely gathered operand still has no matrix, so a scaled contraction over one is refused. It was built once (read the scale at the cell through `cell_read`, no side needed) and dropped in the port when `gather.rs` split into `gather/` upstream; under the split, the values operand stays on the direct path, so this is now only for real gathers. Preserved on `quant-work-backup` |
| 7 | **the metabolic gemv** | the driver. **The routine is written** (`cubek-matmul/src/tiled/quant_gemv/`) and metabolic calls it. See below |

### Item 7, the routine

`cubek-matmul/src/tiled/quant_gemv/`, on `Tiling::over` and the launcher. Four operands and one
verb:

```rust
out.mm_scaled(&w, &x, &scales, Semiring::SUM_PROD);
```

The weight's physical `[d_out, d_in]` buffer is the **lhs**, which is the orientation a decode
step streams: the contraction runs along the buffer's contiguous direction. `K` is spelled
`(KB, KI)`, so one scale per block is the scales operand omitting `KI` — no divisor anywhere in
the launch, and the scales bind at whatever element the checkpoint stored them in.

The plan is metabolic's measured col geometry, restated in the split spelling. A strip of output
rows per cube, a run per plane, and an aligned lane group per row whose lanes interleave the
contraction between them. A group is eight lanes, which is where the fold was measured; how it
reaches eight is the block's business, and that is the one thing the split changed. q4's 32-value
block takes four lanes of `KI`, so the group reaches eight by taking two blocks of `KB`; q8's word
is half as wide, so eight lanes of `KI` cover the block alone and nothing splits `KB`. A cut cuts
one axis or the other and cannot straddle two, so a group wider than a block is *stated* as two
cuts rather than arriving as a stride that happens to cross a scale boundary.

Two things fell out that were not in the plan.

**The device's vector cap is not a refusal.** The routine first inherited `factor >
max_vector_size` from `tests/tile/packed.rs`, which skips q4 on a four-lane device. It is wrong
here for the reason metabolic already recorded on its own selector: the weight binds one `u32`
*word* per line whatever the packing factor expands to. Removed, and q4 runs correctly on this
Metal device — measured, not reasoned about. The activation's eight-wide `f16` line comes back as
two adjacent `vec4<f16>` loads, which is the cap doing its job rather than refusing.

**The routine serves more than one activation row.** `N` is sequential at every level, so a lane
holds that many partials against the weight line it already read. The metabolic arm it replaces
declines a multi-row call; this one does not.

`tests/tiled/quant_gemv.rs` runs it end to end against a host reference, and
`tests/tile/decode_gemv.rs` pins the two shapes the *engine* offers and why the routine takes the
one it does — see the gap below.

### Item 7, the metabolic side

The port estimate in the previous session's notes — "~71 errors plus two `[patch]` blocks" — was
stale. What it actually costs, against `metabolic` at `f3da4b2d`:

- **Three `[patch]` blocks**, not two: cubecl, cubek, *and* burn. cubek needs a cubecl three
  commits past burn's pin (the runtimes refactor, cubecl #1568 / cubek #581), and burn's pin does
  not compile against it.
- **Four lines in `burn-cubecl/src/backend.rs`.** That refactor turned `memory_report()`,
  `memory_usage()` and `memory_persistent_allocation()` from `Result` to plain values and dropped
  `InstallMemoryPoolsError::StreamUnavailable`. Nothing conceptual.
- **Ten identical call sites in `metabolic-extension`.** `Launcher::vector_size` takes a
  `&Geometry` rather than a `&TensorBinding` (cubek #571), so each is `&Geometry::from(&binding)`.

That is the whole port. `cubek-resample`'s removal costs nothing, because the patched `cubek`
umbrella no longer names it.

The arm itself is `matmul/gemv/quant/launch.rs::launch_scales_as_operand`, behind a runtime
switch so both can be measured in one process. It declines silently — a two-level scheme or a
minifloat scale dtype keeps the old arm — and a silent decline read as a fast arm is exactly the
failure mode a measurement invites, so the switch is paired with a launch counter and both the
correctness test and the probe assert on it. That counter earned its place immediately: the arm
was deriving its `QuantValue` from a bit count, which reads `Q8F` as `Q8S` — the same width, a
different range, and no error anywhere.

### Item 7, measured

`gemv_col_quant_bandwidth::scales_as_operand_against_the_widening_arm`, on an M2 Pro: a decode
step's packed projections (36 layers x 4) run three ways per round, order rotated, device-side
`client.profile` on the raw backend. The third variant is the control — the shipping arm over
**f32** scales pays no widening and is untouched by the change, so its own spread says whether
the round is readable.

| | widen f16 | operand f16 | control f32 | delta |
|---|---|---|---|---|
| control spread 0.2% | 46.68 ms | 36.83 ms | 38.23 ms | **-9.85 ms, -21.1%** |
| control spread 3.9% | 48.82 | 36.95 | 38.38 | -11.87 ms, -24.3% |
| control spread 12.9% | 55.43 | 39.99 | 40.72 | *invalid* |

The first row is the number; the box heats across back-to-back runs and the third invalidates
itself by its own control. All three agree on direction and rough size.

**The 9.85 ms is two effects, and the control separates them.**

- **~8.5 ms is the widening pass.** The f16 arm and the f32 arm run the *same* gemv kernel; the
  only difference is the per-launch cast dispatch, and it is 46.68 against 38.23. That brackets
  the ~7.9 ms this plan carried as an estimate, from measurement rather than from reading code.
- **~1.4 ms is the gemv itself**, reading half the scale bytes: 36.83 against the control's
  38.23. The scales are 434 MB of the step at f16 against 868 MB at f32, and 434 MB at this
  box's ~180 GB/s is ~2.4 ms — the same order, so the saving sits where it should.

So the open question — whether the gemv is *also* faster, or only shorter by a dispatch — is
answered: both, and the smaller half is the one that is bandwidth.

One asymmetry worth keeping. The operand arm barely drifts across the three runs (36.83, 36.95,
39.99) where the widening arm drifts hard (46.68, 48.82, 55.43). A kernel whose extra cost is
launch-bound is what heats up that way, which is a second reading of the same fact: the widening
is a dispatch per projection, not bytes.

### Item 2, and what it turned out to be

A contraction's shape was read off the accumulator's *last two axes*. Half of that is not a guess:
the innermost axis is a column edge by construction, because it is the axis the sink lines along.
The other half is, and it was the half refusing a split `N` — `(M, NB, NI)` takes the block index
for a row and leaves the contraction reading a matrix that is not there.

So the column group *reaches*, and how far is read off the operands:

```rust
fn col_split(acc: &Space, lhs: &Space) -> usize {
    let mut split = acc.rank() - 1;
    while split > 1 && !lhs.contains(acc.axis_at(split - 1)) {
        split -= 1;
    }
    split
}
```

An axis the lhs spans stops the run, because an axis the lhs varies over has to be walked against
the lhs rather than folded into a column. `mr`, `cols` and `batch_extents` all read off that one
number. **Landed**, suite unchanged.

The rule this replaced — rows are the accumulator's axes the lhs spans, columns the ones the rhs
does, batch the ones both do — is wrong, and two shapes say so. A conv accumulator shares `N`, `OH`
and `OW` with its image, so every lhs-spanned axis becoming a row would make one `N*OH*OW` matrix
out of a window that is not contiguous; today `N` and `OH` are batch and only `OW` is the row edge,
and which of the two it is is a *tiling choice*, not a fact about membership. And a resampling lhs
can span `COL` (`tests/tile/separable.rs`, an lhs spelled `[ROW, TAP, COL]`), which would take the
accumulator's own column for a row. Membership answers how far the column group reaches. It does
not answer where the rows stop and the batch begins, and nothing needs it to.

**Done.** Every step of it, and the scales are served as lines.

1. **The accumulator's own view** takes its `MatrixAxes` instead of assuming the trailing pair.
2. **Split `N` end to end** (`tests/tile/blocked.rs`), which turned up three refusals: both
   `matrix_mut`s asked whether a projection was *direct* where the read side had already been
   relaxed to ask whether it *overlaps*; `MemData::matrix_mut` built a plain matrix layout where
   its read twin built a projected one; and `scale_side` read the accumulator's columns off its
   last axis. It also turned up an engine bug with nothing to do with quantization:
   `AxisProjection` multiplied line-addressed axes by scalar coefficients, which no single-axis
   column group could show.
3. **The divisors are gone.** `over(bn)` is out of every scales spec and
   `check_lines_hold_one_scale` with it; a scales operand that divides is refused outright.
4. **The scales are served as lines.** Their matrix counts its columns in *blocks* where the
   values' counts lines, which frees them to span only what they vary over and so leaves their
   innermost axis one they do. `ScaledLines` folds them at the leaf rather than under a view,
   because which lane of a scale line a value line takes is a constant only the caller knows:
   `run` is that line's ordinal, and `Lines::lanes` is how the block knows to walk its columns
   under one. A wide scale is refused where the ordinal is not constant — the lhs's columns are the
   contraction, whose step is a runtime index.

`tests/tile/blocked.rs::scales_are_served_several_at_a_time` pins it, and the generated kernel
shows what it is for: one `vec4<f32>` load at one address, four constant lane extracts, where there
were eight scalar loads.

**The register accumulator serves them too.** A promoted block is sized by the accumulator's own
edges and drained through them, and the drain was still taking the trailing pair: with `N` split
the block was `4x8` while the sink view it wrote through was `NB x NI`, so everything past the
first column block was masked away and half the answer landed. `RegisterData` carries the matrix it
was allocated against rather than re-deriving one, since a block that drains through a different
grouping writes its lines where the sink reads something else.
`a_promoted_accumulator_spans_a_split_output_axis` pins the shape with no scales at all, and
`a_promoted_accumulator_takes_scales_by_the_line` pins the whole thing.

`partition_grid` still reads the trailing pair. Nothing reaches it yet: a level that cuts nothing
is dropped, so an unpartitioned promoted walk never asks. A promoted accumulator whose *levels*
cut a split column group would, and that is where to look next.

### Measured

`tests/tile/dequant.rs`'s kernel, timed back to back in one process against a plain `f32` copy of
the same output size (an ad-hoc harness, since a permanent one belongs in an `eval` category and
`cubek-tile` has none):

| | ms/pass | GB/s |
|---|---|---|
| dequant 4096x4096 q4 -> f32 | 0.61 – 1.11 | 70 – 128 |
| copy 4096x4096 f32 | 0.92 – 1.52 | 88 – 146 |

The spread is the machine, not the kernels: four back-to-back runs moved the copy's own number by
40%, which is the thermal variance this box is known for. Only the within-run pairing says
anything, and there the decode tracks the copy across every run while moving an eighth of the read
traffic and taking less wall time for it.

**So the decode is bandwidth-bound, and its arithmetic is not what limits it.** That retires a
suspicion rather than confirming one: `unpack_line` builds its output vector a lane at a time and
the emitted WGSL rebuilds the whole vector per lane, which reads like waste — ten rebuilds per
eight values. It is not worth attacking. `Vector::insert` lowers to a `CompositeInsertOp` with no
assemble-from-components alternative to reach for, the rebuild is a register shuffle a downstream
compiler is free to remove, and the effect being hunted is smaller than the noise floor.

**Out of scope.** The fragment path (`MatrixAxes::whole`, `plane.rs`). A cmma fragment's `16x16` is
a hardware number, so grouping it by extent is right there and stays.

### The staged spelling, probed

The count a scale line covers is a *binding width* today, reconciled against the walk at the leaf
(`FoldRun`, `lines_per_scale`, a divisibility assert). It should be a **cut**, stated where the
level is:

```rust
Tiling::over((values, scales, rhs, out), &[(M, m), (NB, blocks), (NI, bn), (K, k)])
    .level(order, buffering, |cuts, o| {
        cuts.axis(NB, Cut::sequential(8));   // this unit takes 8 column blocks
        o.1.stage(Residence::Register);      // and reads its 8 scales here, once
    })
```

**The DSL already accepts that.** A four-operand `Tiling::over` with the scales spanning `[M, KB]`,
the axis cut, and the residence stated builds, launches, and computes correct numbers;
`OperandSet` already goes to four and no ternary ring is needed, because an operand stages through
its own stage plan rather than the walk's ring.

**What it does not do is honour it.** Stating the residence emits a fill (the kernel grows from 450
to 490 lines and gains a loop), and the contraction reads the scales from global memory anyway —
byte for byte the same four reads at the same loop depth as without it. The staged copy is filled
and never read.

So the whole remaining job is one thing: **the contraction reads the scales from the tile the level
staged.** Everything upstream of that already works, and `FoldRun`, `folded_lane_walk`,
`lines_per_scale` and the divisibility assert all come out when it lands.

### Item 4, surveyed

Three groups, and only the first is a port.

**Ports as it stands** — the register leaf reading its operand in place, one scale level, block or
per-tensor:
`register_matmul_quant_packed_q8` / `_q4`, `register_matmul_quant_rhs_packed_q8` / `_q4`,
`register_matmul_quant_native_block_m`, `register_matmul_quant_native_direct_serve`, the three
`register_matmul_quant_rhs_*_gemv*`, and every `copy_quantized_*` whose point is the *unpack*
rather than the scale. `tests/tile/packed.rs` already carries the lhs and rhs shapes of this in the
new spelling, plus the native one.

**The native store needs nothing at all.** `an_i8_operand_contracts_against_its_scales`: bind the
`i8` tensor as `i8`, `tile()` it, contract. The block casts each value into the accumulator's
element the way it always has, so a store whose element carries no fields never needs a packing
*stated* — a value is whatever its tensor holds, for the same reason a scale is.

**Needs a decision first** (each is a design question, not a copy):

| what | why it does not port |
|---|---|
| every `cmma_matmul_quant_*` | the fragment is loaded from a *staged* operand, so the scale has to fold in at the fill: decode-into-smem, which is Deferred, plus a fill that scales. `mm_scaled` refuses a fragment accumulator on purpose |
| `*_staged_dequantized_smem` | same: a stage holding dequantized values is a *scaled copy*, and `copy_from`'s arithmetic is on the deletion list. So either the verb comes back under its own name, or the stage stays packed and the scale folds at the contraction |
| `copy_quantized_two_level_*` | two scale levels are two scale operands; `mm_scaled` takes one. Either the verb takes a second, or the global scale multiplies into the block scales before the launch (which is what a per-tensor level *is*) |
| `copy_quantized_lookup_*` | a lookup scheme decodes a field through a `2^bits` table. `Packing` names a field's width and sign, not a table; a table is a third operand |
| `copy_quantized_subword_*` | the served line is *narrower* than a stored word. A stated packing ties the served width to `bound_width × factor`, so it cannot spell it. Only the staged fill needs it (`scan_words`) |

The deletions in item 4 are gated on these: `flat()`'s dequantizing read and `copy_from`'s
arithmetic are exactly what the second and third rows still use.

## Deferred

- **Decode into smem** (`dequantize_from` under a hand-written fill). Costs the walk loop, ~8
  lines, and buys nothing until a routine has reuse to amortize it. `mm_scaled` is decode-at-read,
  which is what a decode gemv wants.
- **Scaled MMA (mxfp4/nvfp4).** The instruction eats the format; scales route to the *fragment*,
  not the view. A different rung — and `mma_leaf_scaled` refuses a fragment accumulator today
  precisely so the two do not get confused.
- **The scale factored out of the region.** Where a region sits inside one block, the scale
  multiplies that region's whole contraction once instead of every step — llama.cpp's shape. Valid
  for K-blocks only when the cut divides them, and it changes what a region commits, so it is a
  second contraction shape rather than a tuning knob.
- **Computing the scale in-kernel.** Needs a block reduction before writing, so it belongs at
  drain on the accumulator, not on a view.

## Known gaps

- **A promoted accumulator refuses a `K`-lined rhs**, and that is what keeps the decode gemv's
  partials in memory. `RegisterData::mma_scaled` asserts two things: the block's line width
  equals the rhs's served width, and the rhs's innermost axis is not one the lhs spans. A
  promoted block lines its cells along the *accumulator*, and the col gemv's accumulator is
  `[M, N]` at one activation row, so its column edge cannot be `factor` wide; the activation is
  lined along `K` instead, which is the second refusal. Both are correct as stated — the two
  shapes are a trade, not a ladder, and `tests/tile/decode_gemv.rs` runs each — but the
  memory-backed one is the shape the routine ships on, and metabolic's own note puts the promoted
  accumulator at roughly a fifth of this kernel. Closing it means a promoted block that folds a
  `K`-lined step into one cell, which is `block::contract`'s `step_served` shape rather than a
  new assert.

- **The scales are read one at a time in this orientation**, so item 1b's win does not arrive
  here. A lane owns `rows_per_lane` output rows and one block, and those scales sit a row apart in
  the scales buffer — not a line. Widening them would need a lane to own consecutive *blocks*,
  which is the opposite of the interleave the fold wants. The generated kernel confirms it: one
  scalar `f16` load, broadcast into the weight's `vec4<f32>`. The f16 read is still the whole
  point (half the scale bytes, and no widening pass), but "several per load" is a fact about the
  row orientation, not about the engine.

- **Three tests fail on the CPU backend and pass on wgpu-msl**, none caused by any of this:
  `register_matmul_lane_group_fold` and its promoted twin came with #559 and want a plane wider
  than one lane; `an_i8_operand_contracts_against_its_scales` reads `-70` back as `186`, an `i8`
  read as unsigned, and reproduces identically with the *unsplit* spelling. Verify a CPU run
  against these three rather than against zero.

- The straddle check is a comptime assert *inside* the kernel, so it lands on a worker thread and
  the launch returns zeros beside it. Every other contract assert is the same, but a launch-side
  check would read as a rejection; it needs a routine that states the block.
- A packed operand cannot be *staged in its packed form*: `smem_stored` keys the stored stage on
  the scheme, so a stated packing stages unpacked (correct, just larger). Wanted where a routine
  has reuse to amortize; see Deferred.
- `Packing::Native` cannot be *stated* on a spec: the unpacking view is `u32`-only, and the
  refusal says what to do instead (bind the tensor at its own element and let the contraction cast
  it, which is what `an_i8_operand_contracts_against_its_scales` does). It stays reachable through
  a scheme, which is the only thing that mints it.
- The old machinery is untouched and still shipping. Both spellings compile; nothing is deleted
  until item 4.

## The scale chain: making depth sayable, and clean

The leaf recursion landed (`4342c6f4`): the scales are read through `Lines`, the same trait the
values are, so a scaled operand's scales may carry scales of their own. What did not land is a way
to *say* so, and three debts came with it.

- `FoldRun::compose` states the rule for two levels and nothing can reach its second branch. Every
  call site still builds `ScaledLines<.., MatrixView<..>>`, so `above` is always `FoldRun::ONE`.
- What a level *is* — the scales' matrix axes, the edge they share with the values, the width that
  edge is served at, the lines one scale covers — is derived twice, in `direct.rs` and in
  `promoted.rs`. The two have already drifted: one takes the edge from `shape.reduce_edge()`, the
  other from `operands.contracting(&out)`.
- What a level may be *served at* is two asserts in two files: `sw == 1 || contracted_per_step == 1`
  in the memory nest, `sw == 1 || side == ScaleSide::Rhs` in the promoted block. One rule, two
  exceptions, and neither says what the rule is.
- One test covers a wide scale line, on one of the two accumulators.

### The rule this follows

A scale is a tile that spans fewer axes than the values it multiplies. That sentence is recursive:
the scales are a tile, so they take scales the same way, and one scale for a tile of values makes a
tile of scales for a tile of tiles of values. Two things follow, and the phases below are only these
two applied.

**Nothing in the engine counts levels.** How many there are is a fact about the type the kernel
wrote, never a number the engine holds.

**No call site re-derives what a level is.** A duplicated decision is the bug, whatever the copies
currently agree on.

### Phase 1: one owner for what a level is

`ScaleLevel` — one level of the hierarchy, against the tile below it.

```rust
/// One level of a scale hierarchy, against the values it covers.
pub(crate) struct ScaleLevel {
    /// The scales' own matrix, as the level below reads it.
    axes: MatrixAxes,
    /// Lines of the level below that one scale covers, along the edge they share.
    lines_per_scale: usize,
    /// Scales one read of them serves.
    lanes: usize,
}
```

Built by one function, which reads the count off the axes the way both copies already do: the scale
is constant along every edge axis it does not distinguish, so one read serves every position of
them, and nothing divides an extent.

What stays with each accumulator is what genuinely differs: which edge it walks, and the
`ScaleSide` it reads off the axes. What a level *is* stops being two derivations.

*Proves it:* the ten `scaled` tests, unchanged. Any drift between the two derivations surfaces as a
failure rather than as a difference nobody compares.

### Phase 2: one statement of the servable width

The rule the two asserts are both circling: *a scale line wider than one scale needs each value
line's ordinal along the shared edge to be a constant.* Whether it is, is a fact about how the
caller walks — so the caller states it, and the rule lives once.

```rust
/// Whether the caller walks the shared edge under an ordinal it knows at comptime.
pub(crate) enum EdgeOrdinal {
    /// Each line's position along the edge is a constant, so a lane of a wide scale read is
    /// addressable and the scales may be served several at a time.
    Constant,
    /// The edge is stepped at runtime, so only a scalar read is addressable.
    Runtime,
}
```

Each accumulator answers for itself, once: the memory nest is `Constant` where a step folds one
contracted value, the promoted block where the scales ride the rhs. `ScaleLevel` takes the answer
and refuses a wide binding under `Runtime`. No predicate anywhere — the enum is the state, and the
refusal reads off it rather than off a `bool` each site reconstructs.

*Proves it:* `rhs_scales_are_served_several_at_a_time` on the promoted accumulator, which today's
assert permits and nothing exercises.

### Phase 3: depth becomes sayable

```rust
out.mm_scaled(&w, &x, &blocks.scale(&global), Semiring::SUM_PROD);
```

```rust
/// A scales tile that itself carries scales. Its own `.scale()` returns another, so depth is
/// however many times the kernel said it.
pub struct Scaled<'a, S: Numeric, Above: ScaleOperand> { tile: &'a Tile<S>, above: Above }

/// What the leaf asks of a scales operand: its space, and the reader for everything above it.
#[cube]
pub trait ScaleOperand: CubeType {
    type Above: Lines;
    fn space(&self) -> comptime_type!(Space);
    fn above(&self, #[comptime] levels: Vec<ScaleLevel>) -> Self::Above;
}
```

`Tile<S>` implements it with nothing above; `Scaled` implements it by wrapping one. `mm_scaled`
becomes generic over `ScaleOperand`. The leaf builds level 0 exactly as it does now, with `size!`
minting the binding's width locally, and asks the operand for the rest.

**Levels above the first are served scalar, and that is a property of the hierarchy rather than a
concession.** One line of a level-1 scale already covers `lines_per_scale × lanes` lines of level 0;
a wide read of it fetches scales that will not be wanted for many iterations, buys no bandwidth, and
costs the caller a longer constant-ordinal run to unroll. It is also what makes the reader type
nameable outside the leaf, where `size!` does not reach. The constraint and the right answer are the
same answer.

This is what gives `FoldRun::compose` a caller that reaches its second branch. If the phase slips,
the branch comes out rather than waiting for one.

*Proves it:* `a_scale_of_scales_folds_both_levels_in` — block scales under a per-tensor factor,
`nvfp4`'s own shape, against a reference that multiplies both. And the per-tensor level, spanning no
axis, must leave the walk untouched: the golden is the one-level kernel plus one multiply.

### Phase 4: `mm_scaled` disappears

```rust
out.mm(&w.scale(&blocks.scale(&global)), &x, Semiring::SUM_PROD);
```

One verb again. Reachable now and not before: the ring already carries three operands
(`ed9f9f9e`), so a scaled lhs unwraps to values-plus-chain and takes a slot that exists.
`mma_scaled`, `mma_leaf_scaled` and `MmaScaledWalk` fold back into their plain twins, and
`contract_scaled` becomes `contract` reading a `Lines` that happens to scale.

Stated rather than assumed: `mm`'s walk touches its operands for regions, staging and the op space,
so the operand trait has to carry those too. This is the phase to stop at if it grows. Phases 1-3
stand alone and leave nothing speculative behind.

### Out of scope, deliberately

- The fragment accumulator. A scaled contraction there is a different instruction, not this one
  under a flag; see **Deferred**.
- The N-D gather nest, whose step has no single scalar `k` to address a scale with.
- The decode gemv reading its scales one at a time. That is a fact about the row orientation's
  layout, not about the engine; see **Known gaps**.

### The rule for reviewing this

**No phase may add a number.** If a phase lands and something in the engine says `2`, or asks how
many levels there are, or derives what a level is at more than one call site, it is not done.

### Phases 1 and 2: landed

`e9128d87` and `bb7b22fa`, with `184cbff6` under them.

`ScaleLevel::of` reads what a level is, once. The two derivations it replaced had already drifted —
one took the shared edge from `shape.reduce_edge()`, the other rebuilt it from
`operands.contracting(&out)` — and nothing compared them. `ContractEdges` is what each accumulator
states about its own geometry, which genuinely differs, and it decides nothing.

`EdgeOrdinal` is the one statement of the width scales may be served at. The two asserts it replaced
said the same rule with different exceptions and neither said the rule; `Runtime` carries what about
a walk makes it so, so the message did not get vaguer for being shared. With it came
`rhs_scales_are_served_several_at_a_time`, the promoted block's half of that rule, which nothing had
exercised.

`ScaledLines` now takes a [`Lines`] on **both** sides, so one type serves every level: the values are
an operand's own lines at the level nearest them, and the level below's scales at any level above.

### Phase 3: two findings, and it does not start where this plan said

**The lifetime is not a problem, which was not obvious.** A chain's reader type has to be nameable
outside the leaf, and the first attempt tied it to a borrow of the chain — which needs either a
generic associated type or a higher-ranked bound, neither of which is a safe bet through `#[cube]`.
It is avoidable: an upper level **owns its tile and builds its view inside the read**, so no borrow
escapes, the associated type is plain, and a chain can be stored in a walk and staged like any other
operand. `PlainScales` on `wip/scale-chain-and-broadcast` is that, and `ScaleChain` compiles over it
— `Tile` a chain of one, `Scaled` one more, `Tile::scale` building it.

**But phase 3 cannot start at the surface, because its base case does not work.** A scales operand
spanning *no* axis — the per-tensor level, the base of the hierarchy, and the second level of
`nvfp4` — is refused by the engine in two places, and refused the worst way:

- `Projection::carried_groups` identifies a physical axis by its first term and indexes an empty
  term list. Naming the state it had no word for (`Addressed::Broadcast`, on the wip branch) fixes
  that one.
- `Tile::of`'s `top_window` then computes `rank - 1` on a rank it derives as zero.

Both panic on a worker thread, and the launch returns **zeros** beside them. A per-tensor scale
therefore reads today as a fast wrong answer, not as an error — the failure mode this whole design
exists to avoid.

So phase 3's order is:

1. ~~**A broadcast operand works, and is tested.**~~ **Done.** `a_scale_over_no_axis_covers_everything`
   passes. A physical axis carrying no logical one is its own coordinate group — it shares one with
   nothing, and it is still a buffer axis, so dropping it lost a dimension the layout had to
   describe. That was the second panic; the first was reading a group's identity off an empty term
   list.
2. Then the surface — **and it does not work as written above.** The chain threads through every
   scaled signature and the library compiles (`wip/scale-chain-threaded`), but kernels do not:
   cubecl's `CubeType::ExpandType` has no inverse, so from a `&TileExpand<S>` argument nothing can
   solve `Ch = Tile<S>` and every call site has to name the chain itself.

   ```rust
   c.mm_scaled::<E, E, Scaled<S, Tile<G>>>(&a, &b, &blocks.scale(&global), ..)
   ```

   That states the depth twice — once in the type, once in the value — and reads worse than the
   arity it replaces. **Type-carried depth is right for the leaf and wrong for the surface.** The
   way out worth trying first: let the *kernel* name the chain once, as a type parameter of its
   own, and have the launch pick it, so the turbofish sits where types are already spelled rather
   than at every verb. Decide this before threading anything further.
3. Then each level's coverage in the units its consumer reads — `ScaleLevel::of` measures in value
   lines, and a level above the first is read in the level below's *scale* lines. This is the piece
   with real subtlety left in it, and it is why the surface was not landed blind.

Nothing speculative was left on `cubek-paul`: the chain compiles but is unwired, so it sits on
`wip/scale-chain-and-broadcast` until step 1 makes it safe to land.

## The scale list: the plan that replaces the chain

Supersedes the surface described above.

### What changed, in three lines

`ScaleChain` is dead: `CubeType::ExpandType` has no inverse, so nothing solves `Ch = Tile<S>` from a
`&TileExpand<S>` argument and every call site would state the depth twice. It is replaced by an
ordered `Sequence` of scale tiles, which is concrete at the call site. Order is preserved and no
algebra is assumed: a verb says what each level does, and what the engine may do follows from it.

### Step 0 — confirm the row mapping

Read `Cut::unit(rows_per_lane, Spread::Contiguous, groups)` and confirm group `g` owns rows
`[g * rows_per_lane, ..)`. Every address argument in step 11 assumes it, taken off the name.

**Proves:** nothing yet. It is a prerequisite, and it is minutes.

### Step 1 — `FoldRun` becomes `Reuse`

`instruction/registers/lines.rs`.

```rust
/// How a loaded value is reused across the walk.
pub struct Reuse {
    /// Values one read brings back. Past one, which of them a step takes has to be a constant,
    /// so the caller unrolls: picking a lane of a read is not addressable at runtime.
    pub per_load: usize,
    /// Steps one value serves before the next is wanted.
    pub steps: usize,
}
```

`fold_run()` becomes `reuse()`, which is a question rather than a noun nobody can read.

**Proves:** the 12 `scaled` tests, unchanged. Pure rename.

### Step 2 — `tile_as::<O>`

`physical/arg.rs`, beside `tile_packed::<O>`.

```rust
/// [`tile`](Self::tile) served at a stated element rather than the binding's, cast at the read.
/// Levels of a scale list have different storage and must share one list, so the served type is
/// stated at the call, as it is for a packed operand.
pub fn tile_as<O: Numeric>(&self, #[comptime] space: Space) -> Tile<O>
```

**Proves:** a `ue4m3`/`f16` buffer served as `f32` reads back the same values as an `f32` buffer.

### Step 3 — scales become a list

`ops/matmul/lower.rs`, `instruction/registers/contract/`.

```rust
pub enum Apply { Product }                 // the verb, named not assumed
pub enum Fold  { Combined, InTurn }        // what the verbs license, read off them

fn mm_scaled<Lhs: Numeric, Rhs: Numeric>(.., scales: &Sequence<Tile<ES>>, ..)
```

Applied innermost first, in push order. `Fold::Combined` merges the levels at their own loads only
because every `Apply` is `Product`; `Fold::InTurn` applies them one at a time otherwise. Lifting the
*loads* needs no verb: a level does not change along the axes it omits.

**Proves:** `two_levels_fold_in_order` — block scales under a per-tensor factor on q4, against a
reference that multiplies both.
**Deletes:** nothing yet.

### Step 4 — `per_load` comes from the cut

```rust
let lanes = scales.vector_size();     // before: a width the binding guessed
let reuse = level.reuse_under(cut);   // after: the walk sizes it
```

**Proves:** the two wide-scale tests keep passing with no binding width stated.
**Deletes:** `EdgeOrdinal`, whose only job is reconciling that guess.

### Step 5 — a level against the tile below it

```rust
fn of(scales: &Space, edges: &ContractEdges, side: ScaleSide, ..)   // before: mr/kc/cols, level 0 only
fn of(level: &Space, below: &Space, walk: &Space, apply: Apply)     // after
```

so level 1 against level 0 is the same call as level 0 against the values.

**Proves:** the levels of a list are built by one `map`, with no arm that knows the depth.
**Deletes:** `ContractEdges`.

### Step 6 — nvfp4 mimic, tier 1: the structure

e2m1 values, block scales at **f16**, block 16, per-tensor f32. Same axes, same order, same two
levels; only the block dtype differs from nvfp4. `inside_lanes = 16/8 = 2`, which the blueprint
admits.

**Proves:** `nvfp4_shaped_decode_gemv` against `e2m1_decode(code) × block × global`.

### Step 7 — the claims, read off the kernel

Golden the emitted source and assert two things the clock cannot show clearly:

1. the global appears as **one** load, outside every loop
2. there is **one** multiply per value, not two

**Proves:** the load-placement story. If either fails, steps 3-5 are wrong regardless of timings.

### Step 8 — nvfp4 mimic, tier 2: the bytes

Store block scales as `u8`, decode `e4m3` in software at the scale load — what metabolic already does
for e2m1 *values*, and cheap here for the same reason the design is cheap: once per load, not per
value.

**Proves:** true traffic, one byte per scale, correct range. Likely how it ships on any backend
without native ue4m3, so not throwaway.

### Step 9 — measure

Tier 2 against a one-level f16-scale variant, on the qwen3-8b decode shapes the routine is pinned at.
Isolates what a second level costs.

### Step 10 — the scale layout

> **The innermost axis is the fastest axis lanes differ along, and the load width along it is the
> extent one lane owns of that axis.**

| operand | innermost | load width |
|---|---|---|
| weight `[M, KB, KI]` | `KI`, lanes take different words | `factor` values = one `u32` |
| scales `[M, KB]`, omitting `KI` | `M`, groups take different rows | `rows_per_lane` |

The weight row is what the shipped code already does, which is the sanity check. The scales row is
`[KB][M]` rather than `[M][KB]`. With the Q4 blueprint (plane 32, `inside_lanes` 4, `block_lanes` 2,
`groups` 4, `rows_per_lane` 2): today gives four pairs `2 * blocks` apart, the transpose alone gives
stride 2, and the transpose read `rows_per_lane` wide gives one contiguous eight-element run. Only
the second half of the rule turns the layout into a coalesced read.

**Do:** the transpose in metabolic's re-quantization walk. Then derive the rule in the plan and
*check* it at launch; a mismatch is a rejection, never a kernel quietly running at a fraction of its
speed.

**Gate:** measure first. Of the decode gemv's 9.85 ms win, ~8.5 ms was deleting the widening pass and
~1.4 ms the gemv itself; scales are about an eighth of the weight bytes, and the memory-backed
accumulator is nearer a fifth of the kernel. The bigger fish for speed remains the promoted
accumulator taking a `K`-lined rhs (see **Known gaps**).

### Step 11 — tier 3, when the device arrives

Swap the software decode for a native `ue4m3` binding behind `supports_type`. A substitution: layout,
traffic and structure are unchanged.

### Reviewing this

**No phase may add a number.** Unchanged from above.

**No phase may assume an algebra.** If the engine reorders, reassociates or merges levels, something
must have *said* it could. A rewrite licensed by a comment is not licensed.

### What building it changed

Steps 1, 3, 6 and the load placement landed. Three of the remaining steps were wrong as written, and
contact with the code is what said so.

**Step 4 was wrong: the width is a binding choice, and `EdgeOrdinal` stays.** "The walk sizes the
scale line" does not hold. What the axes determine is how many *distinct* scales a region holds
(`edge / lines_per_scale`); how many to read at once is bounded by that but not fixed by it, and a
narrower binding is correct, just more reads. `EdgeOrdinal` answers a different question, which
survives untouched: whether the caller can hold a constant ordinal at all. What the cut really
determines is the *best* width, which is a coalescing statement and belongs with the layout rule.
**Merged into step 10. `EdgeOrdinal` is not deleted.**

**Step 5 largely dissolved.** It existed so a level above the first could compute its own
`ScaleLevel` against the level below. It does not need to: every level is read at the *same* logical
position and resolves it to its own granularity through its own projection, so one `MatrixAxes`
serves the whole list and no level needs geometry of its own. `ContractEdges` stays.

What looked like the remainder of step 5 was giving each level its own `Apply` rather than reading
the innermost's. It is not worth building yet, and the reason is the point: there is nowhere to
*state* a per-level verb. No operand, spec or scheme carries one, so the plumbing would thread a
constant to the same place it already reaches. The verb becomes per-level when something can say it,
and not before.

**Step 7 is not mechanized.** There is no golden harness in this repo, and `CUBECL_DEBUG_LOG` emitted
nothing for this runtime. The claim it was meant to check is instead structural and readable in the
source: `combined_scales` reads each coarser level once, outside `line`, and `line` is
`inner.read(pos) * coarser`. Worth mechanizing when a harness exists; not worth blocking on.

### Where that leaves the plan

Done: 0, 1, 3, 6, and the load placement. Dissolved or merged: 4, 5 (down to a per-level verb).
Open: 2 (`tile_as`, needed only for tier 2, since tier 1 shares one dtype), 8, 9, 10, 11.

**Step 2 also moved.** It is not needed for tier 1, whose levels share a dtype, and it is bigger than
"small": a narrow buffer served wide is `Packing::Native` today, which is `i8`-only, cannot be
stated on a spec, and would need the stored type to reach the read site inside `MemData<T>`. It
belongs with step 8, where the byte counts are the point.

### Steps 4 and 10 collapse into one, and it belongs to the selector

Three questions were running together. They are not the same question and they do not have the same
owner.

| | asks | owner |
|---|---|---|
| correctness | can a step pick one scale out of a batch at all | `EdgeOrdinal`, in the engine |
| **quality** | **does the batch match the tiles the unit owns** | **the selector** |
| layout | are the batch's scales adjacent in memory | model load, once |

**If a unit loads four scales it should be doing the four tiles they cover.** Nothing makes that
true: the width comes from the binding, the tiles come from the cuts, and they are two decisions
that merely have to agree. A plan where they do not still computes the right answer. It is just a
bad kernel, and plan quality is the selector's business.

**So the engine states an invariant instead of a check.** It must be correct for *any* plan the
selector picks: no assert tying the scale width to `rows_per_lane`, no refusal because a plan is
merely wasteful. Reading fewer scales than a unit could use is more reads, not a wrong answer.
`EdgeOrdinal` stays because it guards something else entirely, which is whether the batch can be
indexed at all.

**The layout and the width decouple, which is what makes the selector's job tractable.** With scales
laid `[KB][M]`, the scales for consecutive output rows at one block are adjacent, and that holds for
*any* `rows_per_lane`. So the layout can be baked at model-load time, before any selector runs, and
stays right whatever the selector later chooses. That matters because the layout is the part that
cannot be revisited per shape: it is in the file. Had it depended on `rows_per_lane`, a knob would
have to be fixed at conversion time for every shape and device the weights might ever meet.

What is left coupled is one number: **the scale line width must equal the tiles a unit owns**,
decided per plan, in the blueprint, beside where `rows_per_lane` is already chosen. That is a real
constraint. `rows_per_lane` is picked today for register pressure and would now also drive scale
traffic, so the knob has to know about scales. But it is one knob against one number rather than a
layout decision entangled with everything.

**Step 10, restated.** Two pieces, in this order:

1. `[KB][M]` in metabolic's load-time re-quantization walk. Independent of every later choice.
2. The blueprint derives the scale line width from the tiles a unit owns, and a test asserts the two
   agree for each plan it produces. Not an engine assert: a plan test.

Gated on step 9 as before, for the reason recorded there.
