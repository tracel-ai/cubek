# feat(tile): indexed operands, and the staging fix MoE needs

Branch `feat/virtual-tile-operands`. Clippy clean on `-p cubek-tile --all-targets`.

## Summary

The side-channel design is the right one and it is carried through consistently. `Indirection`
mirrors `QuantInfo` exactly: erased `Box<[u32]>` plus a comptime spec on `MemData`,
`IndexedTileArg`/`IndexedOperand`/`IndexTable` mirroring the `QuantTileArg`/`QuantOperand`/
`Quantization` triple, host-side `validate` with a `Tile::of_indexed` backstop mirroring
`validate_dequant_at`. The parallel structure reads as deliberate, not as copy-paste.

Keeping an indirect operand direct and untiled is what leaves `projection/*`, `GmemLayout`,
`Walk`, `Extents`, `Partitioner` and `tma.rs` out of the diff, and it is the correct seam.
`Space::walk_invariant` taking the index axes is the right fix at the right altitude: the old rule
was "operand does not span a stepped axis implies its window does not move", and routing is
precisely the case where spanning stops implying that.

The original P1 findings all concerned behavior outside the single-level
`EXPERT = Static(1)` case. They are now fixed and covered below; the remaining P2/P3 findings are
unchanged.

---

## P1: completed

All P1 findings are fixed:

- Nested index coordinates are scaled by their descendant fire-level tile count, with a numeric
  two-level regression covering the former routing-table alias.
- Indexed operands built through `Launcher` keep the target-axis boundary armed, with checked
  out-of-range routing exercised through the launcher path.
- Index tables are validated host-side for their fire-level shape, rank, and dense row-major
  layout before kernel code reads their strides or entries. Short, wrong-rank, and padded-stride
  tables each have a construction-time refusal test.

---

## P2: the lane-distribution rule is applied below the fire level

**`crates/cubek-tile/src/tile/mod.rs:527`** (inside `IndirectionSpec::validate`)

`fires` is set at `:516`, and the child produced at that level carries no indirection at all, so
nothing below the fire level reads the table. The index-axis scope check nevertheless runs on
every remaining level.

Consequence: a two-level MoE matmul that distributes `M` across units at the leaf, which is the
normal shape, is refused with a message about lanes resolving divergent origins, by which point
that is no longer true. The moe tests are single-level and do not catch it.

Gate the loop on the value of `fires` at loop entry. Keep the check *at* the fire level:
`advanced_base` genuinely does read the region there, so a per-lane coordinate at that level is
the real hazard the message describes.

---

## P2: `MemData::at` does not guard its own invariant

**`crates/cubek-tile/src/tile/mem.rs:1960`**

`displacement` is computed unconditionally, but only the `proj.is_direct()` branch
(`mem.rs:1990`) consumes it. The gathered branch discards it silently, and would emit a dead
`self.table[...]` load. The invariant is enforced two layers away, in `IndirectionSpec::validate`
and `StridedTileSource::indexed`, while `flat_accumulate` directly above asserts its own analogous
invariant locally.

Add, at the top of `at`:

```rust
comptime!(assert!(
    self.indirection.is_none()
        || (proj.is_direct() && !self.layout.projection.is_tiled()),
    "..."
));
```

Costs nothing and makes the invariant local.

---

## P2: `IndexPolicy` carries no codegen consequence

**`crates/cubek-tile/src/tile/mod.rs:404`**

Nothing reads the policy except the assert at `tile/mem.rs:339`. Masking is driven entirely by
`spec.boundaries`, which `.checked(true)` sets independently. Three consequences:

- `Trusted`'s doc, "no test is emitted and the window is placed wherever the entry says", is false
  whenever `.checked(true)` was also called. The test is emitted and the window is masked.
- The caller has to state one fact twice:
  `.checked(policy == IndexPolicy::Checked).indexed(..., policy)` (`tests/tile/moe.rs:129`).
- `IndexPolicy` is in `IndirectionSpec`'s `Hash`/`Eq`, so `Trusted` and `Checked` are distinct
  kernel identities compiling identical code.

Either make the policy drive the boundary derivation, which is also the fix for the `settled`
issue above and would collapse the two builder calls into one, or drop the enum and read the
policy off `spec.boundaries` at the target position.

---

## P3: smaller

- **Three matches, one question.** `mem.rs:1960`, `:1967` and `:1980` each match on
  `&self.indirection` and re-derive `fires_at`, a fourth time inside `displacement` itself. One
  comptime match yielding `(displacement, indirection, displaced_at)` would say it once.
- **Unsigned wraparound is unremarked.** `mem.rs:2046` does
  `displacement.fcast::<u32>().fmul(unit)` and relies on u32 wraparound for a negative
  displacement. It is correct, but it is the only place in the file that leans on wrapping without
  saying so.
- **Inconsistent visibility.** `Tile::index_axes` is `pub` (`tile/mod.rs:1059`) while its sibling
  `Tile::indirection_target` is `pub(crate)`. Only `staging/fill.rs` calls either.
- **Never-fires is still silent.** `validate` proves a level exists that *would* fire, not that the
  kernel descends to it. A tile read without any `at` reads the entry-0 window with nothing
  asserting. Worth a sentence in `of_indexed`'s doc.

---

## Tests

Coverage of the shape that is exercised is genuinely good. The shuffled-routing and
repeated-expert cases would catch a table-base off-by-one that identity or reversal would cancel,
and the refusal set covers every `validate` rule that has a host-side site. The reasoning recorded
at the bottom of `moe.rs` for the two rules that have no test is sound.

The remaining gap lines up with the P2 lane-scope finding above:

| Missing case | Finding it would catch |
| --- | --- |
| An index axis distributed across units only below the fire level | Lane-scope over-restriction |
