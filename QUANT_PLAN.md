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
| 7 | **the metabolic gemv** | the driver. Its per-token scale-widening pass (~7.9 ms/step of a 75 ms Qwen3-8B decode step) exists only because the engine reads scales at f32; it deletes itself once the gemv is written in this spelling. **Nothing in the engine blocks it any more** — `a_packed_decode_gemv_runs_in_this_spelling` (`tests/tile/packed.rs`) is the whole shape: packed weights read in place, scales as their own operand, `N` across cubes, partials in registers for the whole `K` walk. What is left is the routine and the metabolic side |

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
