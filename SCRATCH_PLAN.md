# The scratch, made configurable

A plane tile's cells sit across the plane's lanes in a layout only the hardware knows. Anything
that must touch them one at a time puts the tile in shared memory first. Three callers want that
today: attention's row rescale, the drain into a folding output, and (on cubek-Ringo) the scaled
fragment load, which calls its window `landing`.

How much is resident at once is a real trade, barriers against bytes, so it is a setting and not a
constant. This says what it looks like.

## Landed

**`fill_from` picks the door that can fold** (`633acffa`). It chose its path on the destination's
shape alone, so a whole, unmasked, plain destination took the straight buffer fill even when its
writes fold, and a folding store has no address. The write mode joins that condition, so such a
destination falls to the layout walk where its adds land through the sink's own call.
`scan_transparent` now states the disjointness it rests on: one line per unit striding by the cube.
`a_copy_into_a_folding_output_adds` proves it, and was checked to fail without the condition.

**The scratch's size is a setting** (`Resident`). Landed as the two endpoints and not a count, for
the reason the trap below gives. `with_scratch` takes it, sizes the window at `cells * slots *
planes`, and hands **every tile its own slot** where the whole partition is resident, so no drain
depends on the order it walks them in. `CmmaData`'s bounce splits into `spill_to_scratch` and
`add_from_scratch` with the barriers lifted out, and `Tile::drained_into` owns them: three per tile
at one tile resident, two for the whole drain otherwise.

metabolic states it as `GemmStrategy::scratch`, pays for it in `smem_bytes` beside the stage, and
races it crossed with the hypercube counts wherever a fragment leaf reads one. The device test pins
both and counts that the whole-grid drain was actually reached, since a budget that refused the
window would narrow the block rather than the residency.

## Two things this cost that the plan did not predict

**A wildcard on a single-value cube-enum match does not compile.** `match &self.tile_kind { X(t) =>
…, _ => panic!() }` fails with `&TileKindExpand: CubeEnum` not satisfied, pointing at the `#[cube]`
on the whole impl block rather than at the arm. Every arm has to be named, as `dense` and
`split_share` already do it. A *tuple* match keeps its wildcard, which is why `copy_cast_from` has
one.

**The window competes with the stage, so it moves the block.** Counting the scratch in `smem_bytes`
is right — cubecl allocates a cube's shared memory for the whole kernel, so a window the epilogue
alone reads is resident from the first instruction — but it means a whole-grid row is refused the
stage a one-tile row affords and comes out with a smaller block. `cube_block`, the rank's estimate,
is asked before a microkernel exists and cannot see that, so
`the_block_a_row_is_ranked_on_is_the_one_it_launches` no longer claims exactness for those rows. The
over-estimate is the safe direction. It also means the whole-grid arm may lose for a reason that has
nothing to do with barriers, which the race will say.

## The shape, as it landed

```rust
/// How much of a plane's accumulator its scratch holds at once.
pub enum Resident {
    /// No scratch. The fragment's own intrinsic does the work.
    None,
    /// One tile. The smallest footprint, three cube-wide barriers per tile drained.
    OneTile,
    /// The plane's whole partition. Two barriers for the drain, at as many times the footprint
    /// as the partition has tiles.
    WholePartition,
}
```

Opened where the accumulator opens, and it keeps the two counts it already took:

```rust
let acc = c_cube
    .cmma_accumulator::<EA, EL>(&a_cube, fragments, Monoid::Sum)
    .with_scratch(resident, planes, lanes);
```

`planes` and `lanes` cannot come from cube builtins, which the first sketch had wrong:
`Shared::new_slice` needs a comptime length and `CUBE_DIM_Y` is a runtime value.

The drain is one verb, and the size alone decides its schedule:

```rust
pub fn drained_into<Out: Numeric>(&self, dest: &Tile<Out>, #[comptime] cells: Level) {
    let together = self.drains_together();
    if comptime!(together) {
        sync_cube();
        for region in … { self.at(&region).spill_to_scratch(); }
        sync_cube();
        for region in … { dest.at(&region).add_from_scratch(&self.at(&region)); }
        sync_cube();
    } else {
        for region in … { dest.at(&region).copy_cast_from(&self.at(&region)); }
    }
}
```

## Why there is no size between the two

A middle size has to hand each tile a slot chosen by whoever drains it, because the tiles outnumber
the slots. The drain walks the partition in its walk's order and the partition indexes its tiles in
its own, and nothing makes those agree, so `i % slots` over one order does not name the same slot as
over the other. At these two sizes a tile's slot is a fact about the tile — its own index, or the
only slot there is — so no drain's order can matter.

`Walk::region(i)` does take an index and a constant index does fold, so the chunked form is
*expressible*. It is the slot assignment that is not safe, not the loop.

## What is left, in order

1. ~~`Scratch`, the sizing, and `CmmaData`'s two halves.~~ Landed as `Resident`.
2. ~~The drain verb, and `matmul/kernel.rs` calling it.~~ Landed as `Tile::drained_into`.
3. `rescale_rows` onto the same verb, which is where the duplicated lane deal goes away: the
   `lane + t * lanes` loop is written out twice today, in `plane.rs` and in `cmma.rs`.
4. ~~metabolic: the setting, the budget, the raced rows.~~ Landed as `GemmStrategy::scratch`.
   Still open there: the printed space table does not show the window, and the race has not been
   run — the whole-grid arm may lose to the smaller block the budget forces on it rather than to
   its barriers.
5. Name the concept once, before cubek-Ringo's `landing` merges and there are two words for one
   window.

## Why it is worth it

Measured on the M2: the k cut across cubes loses 1.8x at 8 rows and 12.9x at 64, and the cost is
neither the atomics nor the cast. Two controls: the penalty does not move with the cube count, and
a 32-bit product that needs no cast is just as slow. What is left is this bounce, at three
cube-wide barriers per fragment and eight fragments to a plane by default.

A partition-wide scratch does not fit an M2 cube at a 16 by 16 instruction (8 KB a slot, eight
planes), so the useful range is small and device-dependent, which is exactly why it is raced rather
than fixed.
