# Phase 0 — the two-column read

Two kernels' levels written both ways, by hand. Nothing here ships. The question is only whether
the right column reads better than the left, and where it does not.

Numbers are one concrete plan: a `16x8x16` instruction, `2x4` fragments a plane, `2x4` planes a
cube, two instruction-steps to a stage.

---

## The gemm, staged

### Today — outermost first, every level a size

```rust
let (block_m, block_n) = microkernel.plane_block();                 // 32, 32
let (stage_m, stage_n) = (planes.m * block_m, planes.n * block_n);  // 64, 128
let stage_k = fragments.depth();                                    // 32

vec![
    Level::cubes(&[(M, 64), (N, 128)]).batches(&[B]),   // cut m x n into 64 x 128
    Level::walk(&[(K, 32)]).filled_by(fillers),         // step k by 32
    Level::planes(&[(M, 32), (N, 32)]),                 // cut 64 x 128 into 32 x 32
    Level::walk(&[(K, 16)]),                            // step 32 by 16
    Level::walk(&[(M, 16), (N, 8)]),                    // cut 32 x 32 into 16 x 8
]
```

Three of those five numbers are products the caller just computed and the engine will divide back
out. `32` appears twice meaning two different things (a plane's block, and a stage's depth).

### Proposed — leaf first, every level a count

```rust
Tiling::leaf(&[(M, 16), (N, 8), (K, 16)])        // the instruction: the device's, not a choice
    .walk(&[(M, 2), (N, 4)])                     // fragments a plane holds
    .walk(&[(K, 2)])                             // instruction steps in one stage
    .planes(&[(M, 2), (N, 4)])                   // planes a cube holds
    .walk(&[(K, Whole)]).filled_by(fillers)      // every stage k holds
    .cubes(Whole).batches(&[B])                  // every box m x n holds
```

**Level for level, in reverse.** Today's list read bottom-up is exactly this list read top-down,
so the port is a reversal and a change of what each number means — not a change of structure.

**What the right column says that the left cannot.** `2x4` fragments and `2x4` planes are visibly
the same kind of number; in the left column they are `16x8 into 32x32` and `32x32 into 64x128`
and you divide to see it. The instruction stops being a level and becomes the leaf, which is what
it is. `plane_block()` and the `stage_m` multiplication have nothing left to compute. And the two
`Whole`s are the only two runtime numbers in the launch, sitting where they belong: the outermost
loops, over the problem's own extent.

**Verdict: better, clearly.** Six lines against eight, no arithmetic, and the one fact the device
dictates is the first thing stated.

---

## Decode attention, unsplit

### Today

```rust
vec![
    Level::cubes(&[Cut::new(QP, 1), Cut::new(KV, 1)]).batches(&[B]),
    Level::walk(&[(S, block)]),
]
```

`G`, `D` and `DV` are named by no level and handed down whole. The leaves take their spread from
`CUBE_DIM_X` rather than from a level — which is why a decode blueprint overrides `cube_dim`
instead of the default reading it back off these levels. **Its own module doc calls that the gap.**

### Proposed

```rust
Tiling::leaf(&[(D, line), (DV, line)])      // one lane's vector of the head dim
    .lanes(&[(D, per_lane), (DV, per_lane)])// a plane covers the head dim
    .walk(&[(S, block)])                    // positions in one block of the cache
    .walk(&[(S, Whole)])                    // every block of the prefix
    .cubes(&[(QP, 1), (KV, 1)]).batches(&[B])
```

**This is where it bites, and the bite is informative.** Bottom-up cannot start without a leaf,
so decode has to state the one it currently omits — and stating it is exactly the gap its doc
names. The proposed column is longer than today's two lines because it says two things today's
does not: what one lane holds, and that a plane covers the head dim between them.

So the port is not mechanical here. Two ways out:

1. **State the leaf.** Longer, and it closes the `cube_dim` override — the default could then read
   the cube back off the levels like every other family.
2. **Allow a leaf that is the cube.** A spelling for "below this, the kernel's own business",
   keeping today's two lines. Cheap, and it keeps the gap.

(1) is the better end state and is a change to decode, not to the DSL. It should not be smuggled
into this refactor; it should be a rung of its own that this refactor makes obvious.

---

## What phase 0 answers

* The gemm's port is a **reversal**, level for level, and the result is shorter and has no
  arithmetic in it. The claim holds where the stack is deepest.
* Decode's port is **not mechanical**, and the reason is a gap already written down. The
  multiplicative form does not create that problem; it refuses to hide it.
* The two `Whole`s are load-bearing and want the good spelling early, not in phase 3 — in both
  kernels they are the outermost levels, and they are the only runtime numbers in the launch.

## What it does not answer

Quantized packing and the gemv's lane folds are untouched here. Phase 2 still has to meet them.
