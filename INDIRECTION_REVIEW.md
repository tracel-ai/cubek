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
`EXPERT = Static(1)` case. The P1 and P2 findings are now fixed and covered below; only the P3
cleanup remains.

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

## P2: completed

All P2 findings are fixed:

- Lane-distribution validation stops after the fire level; the fire level itself remains checked.
  A two-level construction test covers a unit-distributed index axis below resolution.
- `MemData::at` asserts its direct, untiled indirection invariant locally before computing the
  displacement.
- `IndexPolicy` now drives the target-axis boundary derivation. `Checked` arms the target without
  a duplicate `.checked(true)`, while `Trusted` suppresses a generic target mask; both behaviors
  have focused tests and now produce distinct boundary codegen.

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

The P1/P2 gaps identified by this review now have regression coverage: launcher-built checked
routing, multi-level deep-fire addressing, short and malformed tables, post-fire lane
distribution, and policy-driven target boundaries.
