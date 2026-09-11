# Quest 1 · One layout word

A tensor's layout is stated once, on its binding, and every kernel adapts to it
without a match arm of its own. Started 2026-09-10 on metabolic main `7aca10b3`
(quests 3, 5, 6, 9, 10 merged), cubek at the pin `935fdf24`.

## What the tree said

Since quest 9 the packed mystery is three-quarters gone: no `PackedSide`, no
`Packed` delivery, the weight is the rhs in both packings. Five words remain,
and every one is a reading of the same fact, *which axis the weight is
contiguous along*:

| word | where | what it restates |
|---|---|---|
| `Contiguous::{M,K,N,Neither}` | `matmul/key.rs` | `Projection::contiguous(&dims)` collapsed to one axis |
| `WeightLayout::{Row,Col}` | `matmul/gemv/analysis.rs` | `Contiguous::{N,K}` |
| `PackedAlong::{K,N}` | `matmul/scaled/problem.rs` | the same axis, read off the scheme instead of the strides |
| `RhsOrder::{Stated,InBufferOrder}` | `matmul/gemm/blueprint.rs` | "bind over the buffer's own order", allowed only where the leaf can read it |
| `Delivery::{Tma,Strided}` | `matmul/gemm/blueprint.rs` | cubek's `Delivery::{Tma,Copy}` |

The float gemv still binds the problem *transposed* for a col weight
(`is_swapped`, `GemvVectors`, two launch arms, a transposed output) — the very
thing quest 9 removed from the scaled kernel.

## The word

cubek's `Axis`. `Some(K)` is "contiguous along K"; `None` is a strided or
broadcast view. The key holds `Option<Axis>` per operand, a derivation that
already declined the rest holds `Axis`.

## metabolic, in commits

1. **`Contiguous` → the axis.** `MatmulTuneKey { lhs, rhs: Option<Axis> }`,
   `MatmulShape.rhs: Option<Axis>`, `GemmInputs.rhs`, `GemmAnalysis.rhs`. One
   reading, `matmul::key::contiguous(shape, strides, axes) -> Option<Axis>`.
2. **`PackedAlong` deleted.** `ScaledProblem.packed_along: Axis`, read off the
   weight's strides through the same reading; the scheme's `PackedU32(dim)` is
   held to agree in `problem_of`, the one place both are in hand. The launch
   stops matching: the packed axis is split into blocks and inside wherever an
   operand carries it, and an operand lined along it is served a word's values
   at a time.
3. **`WeightLayout` deleted.** `GemvAnalysis.contiguous: Axis`, every rule a
   two-way question of it.
4. **The float gemv stops swapping.** The weight rides the rhs in both
   layouts, bound over the order its buffer steps; the activation is the lhs
   `[M, K]`; the output is straight. `is_swapped`, `GemvVectors` and the
   launch's two arms go; the widths follow from the axis each operand lines
   along. The col spaces cut `N` where they cut `M`. `cube_fold` builds the
   sink's own spec. Verified on Metal: `gemv_settings`, `gemv_plane_fold`,
   `correctness`, and a decode A/B.
5. **`Delivery` is cubek's.** The host enum, `TMA_MAX_BOX_DIM` and the batched
   check go; `affordable` asks `Delivery::validate_tma`.
6. **The book.** `layout.md` names the one word; the glossary and the kernel
   rules follow.

`RhsOrder` stays: it is the one knob the engine still needs, because only the
cmma fragment load reads a transposed stage. Its doc names that as what
deletes it.

## Progress, 2026-09-10

- metabolic `quest1/one-layout-word`, six commits on main `7aca10b3`, unpushed:
  `c5ffe3e2` the key names the axis · `54fd725e` scaled: PackedAlong gone ·
  `44c76dcb` gemv: WeightLayout gone · `8b9e6615` gemv: the weight rides the rhs
  · `a6231b43` gemm: the delivery is cubek's · `61f47223` the book. Green:
  279 host tests, clippy with warnings denied, fmt; on Metal, `gemv_settings`
  (every setting, both layouts, f16 and f32 — the suite is gated on `metal`
  now), `gemv_plane_fold` 16–17 of 17 per run with the failing test moving
  between runs and passing alone, which the system log names as
  `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` while two other
  sessions ran device suites on the same M2.
- cubek `quest1/one-layout-word` (`7bbc7329`): `Axis` derives serde, the TMA
  box limit is public. metabolic's working tree carries a LOCAL-ONLY
  `[patch]` to `../cubek-Paul` as its tip commit until the pin moves; drop it when it does.
- Not exercised on this Mac: the cube-scope gemv (`cube_fold`), whose sink
  now states its own spec, since Metal's fixture carries no shared f32 atomic
  add; and the fused gemv carrier, which is behind the `matmul-epilogue`
  feature that does not compile at this pin.

## cubek

Nothing until a gap shows. Candidates, in the order the metabolic commits
would hit them:

- `Axis` deriving `Serialize`/`Deserialize`, so the tune key carries it with
  no glue (the glue in `key.rs` is the interim).
- A folded step at more than one lhs row (commit 4 at `rows > 1`, and the
  scaled K form's `m > 1` ticket from quest 9).
- The register block and the mma transport reading a stage in the buffer's
  order — what would delete `RhsOrder`.

## Not this quest

Packing as the innermost storage level, a line addressing a word-sized tile,
the blob. Each is an engine design of its own and none is what makes the
metabolic side unreadable today.

## The follow-ups, 2026-09-11

Louis's call on the brainstorm: #2, #3, #4 yes; #1 and #6 are goals, too
risky now; #5 as a storage level is not elegant.

- **#3 landed** (`08ef582d`): the scaled K form and the gemv's row form
  carry every activation row; the accumulators grow with the rows;
  `gemv_settings` and the quantized col-form test check three rows.
- **#2 landed** (`58809862`): `PlaneFold`, `CubeFold`, `KSplit`,
  `UnitPerVector`; one `space()` in the table's shape; the plane fold at the
  plane scope emits the levels it did.
- **#4 landed** (metabolic + cubek `e1c06096`): `StridedTileSource::stored`
  orders the subspace dims by stride and carries the labels; the gemv and the
  gemm bind the view and say `[K, N]`; `physical_weight`, the gemm's
  `in_buffer_order`, `RhsOrder::axes` gone. `RhsOrder` stays as the plan's
  choice between stored and stated, for the leaf reason.
- **#5, the honest version, is blocked by a quest 3 decision.** The space
  could be `M, N, K` with the scales projected as `K / block` — the lane cut
  needs no second granularity, a single word-wide interleave of `K` gives
  every lane the same words. But cubek refuses a scales operand that divides
  (`check_scales_omit_rather_than_divide`, `contract/scale.rs`), by design:
  "spell the block as an axis and omit the position inside it, so one scale
  per block is what the axes say rather than what the arithmetic does." That
  is quest 3's "block level stated" rung. Doing #5 means reverting it and
  moving the line-straddles-a-block check to the launch, where
  `validate_scheme` already states it arithmetically. Louis's call.
