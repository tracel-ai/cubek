# Levels state counts, built from the leaf

A level says **how many of the thing below it**, not how big a piece of the thing above it.
`(M, 128)` becomes "four of the thirty-two a plane holds"; the size is the product and nobody
writes it.

## The argument

**The round trip is already there, and it is pure waste.** metabolic holds counts —
`tiles_per_plane`, `tiles_per_stage`, `planes_per_cube` are all "how many". Every `space.rs`
multiplies them into an edge (`stage_m = planes.m * block_m` → `128`), hands that to the engine,
and the engine divides `1024 / 128` to recover the count it was given. Both halves already think
multiplicatively; only the interface between them is divisive.

**The kernel already compiles multiplicatively.** A gemm launches under `KernelForm::Dynamic`: it
reads the cuts at compile time and never the extents. The sizes it bakes in are products of the
levels — yet a level is written as a division of a problem the kernel cannot see. The spelling
and the semantics disagree, and every reader pays for it.

**Counts force bottom-up.** You cannot say "two of the thing below" before saying what is below.
Multiplicative and innermost-first are one decision, not two. It starts from the only level that
is not a choice — the instruction's shape is the device's — and every line after it is a decision,
in the order the decisions are made. The one unknown, how many cubes the problem needs, lands
last, where it belongs.

## What disappears

* **Divisibility between levels, as a category.** "Do the planes deal into whole tiles", "does the
  hypercube count divide the `K` walk", "is the storage tile a level's tile" are one question
  asked in three places. A count cannot fail to divide, because nothing is divided.
* **`Coverage`'s two variants.** Its own doc says "pin one, derive the other" — that is this
  duality, already surfaced as two spellings of one thing.
* **The edge/extent collision.** `(M, 128)` looks like an extent because it is a size. A count
  never does, so `shared_by(cubes)` beside `(N, cube_n)` stops reading as three kinds of number.
* **A runtime multiply per walk.** `walk.total() * stride` is a product of counts constant by
  construction; only the outermost level's is not.

## What does not go, and the phase-4 gate depends on telling them apart

* **The built tile against the problem's extent.** `ceil(m / cube_m)` at the top, with a ragged
  last tile. A fact about the caller's shape, not about the levels. Raggedness stops appearing at
  every level and appears once, here.
* **`Hypercube::of`'s test.** Whether the cube count divides the `K` walk is a question about two
  counts, and it is what still chooses a cut over a run.
* **Loops stay outermost-first.** `for cube in space { for plane in cube { … } }` cannot be
  written bottom-up: you do not know which region you are in until you have descended. So the
  declaration and the body read in opposite directions, permanently. The bridge is the launch
  table each kernel already prints — top-down, with resolved sizes. Authoring is bottom-up
  because that is how you choose; reading a finished plan is top-down because that is how it runs.
  The builder is the **single** named place the reversal happens.
* **`Walk`, `Region`, `Run`, the walk order, the scope words.** Untouched. This rewrites how a
  partitioning is stated, not what it means.

## The one exception that needs a word

One level always wants to say "all of it" — a reduction's whole `K`, the batch axes. That is a
count nobody knows until launch, and pretending it is a product would be a lie. `Edge::Whole` is
already that marker; bottom-up it becomes the exception rather than one spelling among several.
Do not settle its spelling in advance: phases 1 and 2 make its real shape visible.

## cubek-tile

| Add | Why |
|---|---|
| A tiling builder taking counts from the leaf up | The whole point; phase 1 emits today's `Level`s unchanged |
| A per-axis **count** on a level | One number, and it cannot fail to divide |
| One spelling for "all of it" | The honest exception above |
| `Partitioning::tile()` — the built shape, leaf to top | What the launch compares to the real extent |
| A temporary test: new builder's partitioning equals the old one's | The gate that makes phases 1–2 provably a change of spelling |
| A comptime bound on a run's touched tiles | Only possible once counts are constant (phase 5) |

| Delete | Replaced by |
|---|---|
| `Edge::Cut(usize)` | the count (`Edge::Whole` survives) |
| `Coverage::{Instances, TilesEach}` | one variant |
| `Cut::new(axis, edge)` and its `edge` field | the count |
| `Level::cubes/planes/lanes/walk(&[(axis, size)])` | the leaf-up builder |
| `.instances(n)` / `.tiles_each(t)` | one verb |
| the two arms of `run_length` / `instance_count` | one arm |

## metabolic

Nothing to add: the strategies already hold counts.

| Delete | Why |
|---|---|
| every `stage_m = planes.m * block_m` in each `space.rs` | the counts go straight through |
| gemm: "the plane grid the division settled does not deal into whole tiles" | unrepresentable |
| gemm: "the cubes cutting an output tile's K walk do not deal it evenly" | unrepresentable |
| gemm: `realizes_storage` | the storage tile *is* a level's count |
| the `largest_that_fits(…, is_multiple_of)` snaps | nothing to snap to |
| the resolved-size arithmetic in `cube_edges()` and friends | read off the built tile |

## The phases, and the gate on each

**0 — The two-column read.** Write two kernels' levels both ways by hand, in one scratch file:
the **gemm** (deepest stack) and **decode attention** (omitted axes, forked, awkward). Nothing
ships.
*Gate: read side by side, the bottom-up one is better.* If it is merely different, stop — the rest
is not worth its cost.

**1 — The builder, over today's levels.** Counts from the leaf up, emitting the `Vec<Level>` that
exists now. No engine change, no kernel change.
*Gate: the gemm's rewritten `space.rs` produces a `Partitioning` equal to the one it replaces,
asserted in a test, every suite green.*

**2 — Every other caller.** The gemv, `matmul/scaled`, both attentions, cubek's own kernels and
tests. Same gate each. *Done 2026-09-15 for every production kernel in both repos* (the split
merge's two arms and `PlanePartition::fragment_level` were the stragglers); the ~250 test sites
and the eval harnesses move with phase 4.
*This is where the vocabulary meets contact* — GQA's omitted axis, the gemv's lane folds,
quantized storage tiles. If the product model does not hold somewhere, it must surface here and
not in phase 4.

**3 — Settle "all of it".** With every caller ported, the exception's real shape is visible.
*Settled (2026-09-15), in `tiling.rs`'s module doc and enforced by the builder:*
* It is a level that names an axis and no count: `walk_every`, `cubes`, `batches`. Every caller
  spells it that way and no other; nothing emits `Edge::Whole` on purpose any more.
* "Every" means every tile the level above hands down; the whole axis only at the outermost level
  naming it. Under `cubes(&[.., K]).across(K, n)` the walk below takes the cube's run.
* It closes the axis. A walk above it panics; a cube level may name it again **only** to deal it
  across cubes (`across`), which is checked when the levels are built.
* An axis a level does not name is *not* "all of it"; it is not that level's. `Edge::Whole` stays
  as that marker, distinct from an every-level, and phase 4 keeps them distinct.
* What phase 4 therefore puts in `Level`: per named axis, a tile and a **count**, `Of(n)` or
  `Every`. `Coverage::{Instances, TilesEach}` collapse into `Of(n)` (workers stated, one tile
  each; a walk's `Of(n)` is its steps) and `Every` is the one runtime count.
* The in-kernel view levels (`region.over(&Level::walk(&[(axis, size)]))`, 14 sites) are **leaves**
  — the one size a sub-tiling states — and `fragment_level` was the last one derived by division.
  Phase 4 gives `over` a leaf spelling rather than a level.

**4 — Delete the divisive spelling.** Only when nothing states an edge.
*Gate: the between-level divisibility refusals are **deleted, not ported**, and a test pins that a
shape which used to be refused is now unrepresentable.* A refusal that survives means either the
model is wrong or the refusal was one of the three above that legitimately stays — say which.

**5 — The comptime harvest.** Counts below the top are constants, so trip counts fold. Read the
MSL to see what actually folded, and take the refinements that needed it — first among them the
comptime bound on a stream-K run's touched tiles.

**6 — The tune keys, last and alone.** Settings that were sizes become counts, so names change and
rows re-race. Nothing else in flight when this lands.

## Risks

* **The window where both spellings exist** is phases 1 to 3. Keep it short; the equal-partitioning
  gate is what keeps the two honest while it is open.
* **Attention and quantized packing** are where "a product of counts" is least obviously true.
  Phase 2 exists to find that out early.
* **Every tune key invalidates** at phase 6. A re-race of the whole table, not a bug.
* **This is the DSL's core** plus every kernel in two repos — larger than any quest on the board.
  Phase 0 is the cheap way to find out whether it is worth it.
