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

## The shape

```rust
/// Where a plane tile becomes addressable cells: a window of shared memory, one per plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scratch {
    /// No window. The tile's own intrinsic does the work.
    None,
    /// This many of the plane's tiles resident at once. One is the smallest footprint and a
    /// barrier trio per tile; the partition's own count is one trio for the whole drain.
    Tiles(usize),
}
```

Opened where the accumulator opens, and it keeps the two counts it already takes:

```rust
let acc = c_cube
    .cmma_accumulator::<EA, EL>(&a_cube, fragments, Monoid::Sum)
    .with_scratch(scratch, planes, lanes);
```

`planes` and `lanes` cannot come from cube builtins, which was wrong in the first sketch:
`Shared::new_slice` needs a comptime length and `CUBE_DIM_Y` is a runtime value. They stay.

Sizing: `cells * resident * planes`, and each plane takes the window at
`(UNIT_POS / lanes) * cells * resident`.

## The verb, and the trap

`Walk::region(i)` takes an index and a constant index folds, so a verb can chunk the cells walk:

```rust
#[cube]
pub fn drained<E: Numeric, Out: Numeric>(
    cells: Walk,
    acc: &Tile<E>,
    dest: &mut Tile<Out>,
    #[comptime] resident: usize,
) {
    #[unroll]
    for chunk in 0..comptime!(total.div_ceil(resident)) {
        sync_cube();
        #[unroll]
        for j in 0..resident { acc.at(&cells.region(i(chunk, j))).spill(j) }
        sync_cube();
        #[unroll]
        for j in 0..resident { dest.at(&cells.region(i(chunk, j))).add_from(acc, j) }
        sync_cube();
    }
}
```

**The slot must come from the verb, not from the fragment.** Today `with_scratch` hands every
fragment the same window and `accumulate_cast_window` owns the whole bounce. The obvious extension
is to hand fragment `i` the slot `i % resident` at open time, and that is wrong: the fragment index
runs in the partition's `(mi, ni)` order and the drain runs in the cells walk's order, and nothing
makes those the same. So `CmmaData` carries the plane's whole window and the spill and the add take
a slot index the verb assigns.

The round trip is the same shape with the reload put back, which is `rescale_rows`.

## What is left, in order

1. `Scratch`, the sizing, and `CmmaData`'s two halves taking a slot. No behaviour change at
   `Tiles(1)`.
2. The `drained` verb, and the drain in `matmul/kernel.rs` calling it instead of looping
   `copy_cast_from` per cell.
3. `rescale_rows` onto the same verb, which is where the duplicated lane deal goes away: the
   `lane + t * lanes` loop is written out twice today, in `plane.rs` and in `cmma.rs`.
4. metabolic: a count on `GemmStrategy`, clamped by the shared-memory budget *after* the stage's
   claim, raced by the table, printed in the space table. `MatmulNest::Fragments` drops the
   `Scratch { planes, lanes }` struct for the count.
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
