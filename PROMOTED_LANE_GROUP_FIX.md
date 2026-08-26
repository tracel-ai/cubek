# cubek: the promoted accumulator drops every lane group but the first

Found 2026-08-25, root-caused and fixed 2026-08-26. Against cubek `cff1f01f`,
reproduced on Metal (M2 Pro) from `metabolic-extension`'s float col gemv.
**The fix is written and green on the `cubek-ringo` branch** — this is its PR
text.

**Severity: silent wrong answer.** No comptime assert, no launch refusal, no
panic. The kernel compiles, runs, and writes three quarters zeros.

---

## The space

A gemv in the swapped orientation, so the weight binds as the lhs on `(M, K)`
and the activation as the rhs on `(K, N)`. One instruction level, cut so that a
32-lane plane carries **four output rows at once**: aligned groups of 8 lanes,
one row each, the group's lanes interleaving `K` between them and their partials
folding inside the group.

That is `LaneShare::Group { fold_mask: 7 }` — a *segmented* plane reduction. A
whole-plane fold would be `LaneShare::Plane`, and it is the only folded case any
existing test covers.

## What goes wrong

Under the memory-backed leaf this space is correct at every shape and dtype.
Promote the accumulator and 3 of every 4 output rows are never written:

```text
Position 0: correct
Position 1: 0 != 4.9241385
Position 2: 0 != 6.807028
Position 3: 0 != 6.4836226
Position 4: correct
```

Every index `≡ 0 (mod rows_per_plane)` is right and the rest keep whatever the
launch left there. Not garbage, not a mixed sum — **the right number written to
the wrong count of addresses.**

## Root cause: one odometer, decoded twice, two different answers

The lane index is a mixed-radix odometer over the axes a level distributes. Here
`K` takes 8 instances at weight 1 and `M` takes 4 at weight 8, so `M`'s digit is
`(lane / 8) % 4`.

Two places decode it, and they disagree about **which axis list is the
odometer**:

| | reads | result |
|---|---|---|
| `Space::lane_share` | `self.partitioner.axes()` — every axis of the operation | `fold_mask = 7` ✔ |
| `Walk::from_counts` | `space.axis_at(q)` for `q` in `0..space.rank()` | `M` at weight 1 ✘ |

The accumulator's tile is a **projection**: its space is `{M, N}`, rank 2 —
`K` is contracted away and gone from its axis list. Verified directly, by
panicking inside the descent:

```text
PROBE fragment_window Instance: rank=2 axes=[Axis(0), Axis(1)]
```

So the walk finds no same-scope axis inside `M`, gives it weight 1, and decodes
`lane % 4`. For the four elected writers — lanes 0, 8, 16, 24 — that is
`0, 0, 0, 0`. All four store to row 0; rows 1–3 are never addressed.

The generated Metal shows the two derivations side by side in one kernel:

```c
uint32_t const v92  = v90 % v91;    // line 623: lane % 4   → the store's row
uint32_t const v118 = v90 / v117;   // line 630: lane / 8
uint32_t const v120 = v118 % v91;   //           % 4        → the lhs's row
```

Both paths run the same 3-step `simd_shuffle_xor` butterfly and the same
`lane & 7 == 0` writer election. Only the address differs.

This also explains why the shares behave differently, which is how the bug was
localized: `Plane` means one `M` instance, so there is no `M` digit to get
wrong, and `Whole` does not fold at all. Setting `group_lanes` to the full plane
width makes the identical space pass.

`Partitioner::axes` already documents the invariant the walk breaks:

> The axes this level distributes, **which outlive the space they came from**: a
> level keeps every axis of the operation, so an output space (`{M, N}`) still
> names its contraction.

## The fix

The defect is the second derivation, not the arithmetic — patching `lane % 4`
would leave two decodes free to drift again. So the walk is taught to read the
same list `lane_share` reads.

`Space::inner_weight_unspanned(axis)` — the instance-index weight this space's
own axis list cannot see: the product of the instance counts of the same-scope
axes inside `axis` that the partitioner distributes and this space does not
span. It panics where such an axis has no comptime instance count, rather than
assuming 1, because assuming 1 is exactly this bug.

`Walk::from_counts` then folds both halves into `inner_weight`: the spanned
axes' possibly-runtime counts as before, times the unspanned axes' comptime
product.

Two files, ~45 lines including the doc comments.

**It removes work rather than adding it.** The decode is loop-invariant prologue
arithmetic that folds to constants, and the duplicate is now gone from the
codegen — one `(lane / 8) % 4` feeding both the lhs read and the store, where
before the kernel computed both forms:

```c
uint32_t const v94 = v93 / v90;   // lane / 8
uint32_t const v96 = v94 % v95;   //        % 4   → the lhs read *and* the store
```

## Tests

`crates/cubek-tile/tests/tile/matmul.rs`, a pair sharing one space builder:

- `register_matmul_lane_group_fold` — the memory-backed leaf over the segmented
  fold. The control: it passed before the fix, so a promoted failure is the
  promoted path and not the space.
- `register_matmul_promoted_lane_group_fold` — the same fold promoted. Fails
  3-of-4 before, passes after.

They are written against `plane_size_max`, so the group count follows the
device's plane width rather than hardcoding 32.

Full `cubek-tile` suite on Metal: **459 passed, 0 failed.**

## Two follow-ups this leaves open

1. **`promoted.rs:52` should go.** It is a guard against a case that was never
   the real one, and its comment describes something close enough to this bug to
   have misdirected the first day of investigation. It tests whether the rhs's
   *last* axis is contracted; ours is `N`, so it never fired.
2. **`reduce_axis` drains through the same code**, and its only lane-split test
   (`resident_max_over_lane_split_k`) is a `Plane` share. A segmented reduce was
   very likely broken the same way and is now very likely fixed; either way it
   wants its own test.

## Why metabolic wants it

With the memory accumulator the gemv sums the whole `K` walk in the *output's*
element type, so an f16 model accumulates in f16 across 128 steps per lane —
`gemv_plane_fold.rs:126` checks it at `Tolerance::relative(1e-2)` where the f32
path gets `1e-4`. Measured elsewhere, f16 accumulation costs 9.1e-2 relative
error at `d_in = 4096` against 6e-4 for an f32 sum. The register accumulator is
the only fix the tile API offers, and `Group` is the share every rows-in-flight
gemv plan lands on — the quantized arm included, once its own packed-operand
assert lifts.
