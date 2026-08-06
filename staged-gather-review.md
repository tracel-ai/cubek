# Review: staging a gathered operand

Scope: the `cubek-tile` changes that let a gathered (affine-projected) operand run under
`Schedule::Staged`, resolving its projection at the fill's source read.

Verification run: `cargo check -p cubek-tile --all-targets` and
`cargo clippy -p cubek-tile --all-targets` are clean. Tests were not run.

## Verdict

The design is sound and matches the existing `Tile::nd` read path exactly: same view stack
(`lines_storage` -> `base` -> `window` -> `AxisProjection`), same `check` flag
(`src.access.overhang.masks()`), and `read_checked` composes down through the layers, so an
out-of-range tap still masks to zero.

Two soundness claims were traced and hold:

- `physical_pos` returns a coordinate ordered by the destination's positional projection, which
  for `smem_over` is `Projection::direct_over(space)`. So `space.position(term.axis)` in
  `AxisProjection::to_source_pos` indexes it correctly.
- Storage tiling is not a false positive for `gathered()`: `MemData.projection` is
  `spec.projection.untiled()`, so a `[grid..., tile...]` buffer stays `is_direct()` and never
  reaches `AxisProjection`, which has no digit splitting.

## Findings

### 1. Unasserted contract: `space` is the destination's, used to address the source

`crates/cubek-tile/src/tile/mem.rs`, `fill_from`.

The source's `AxisProjection` is built from `dst.space`. In the staging path the two provably
coincide (`smem_like` uses `operand.space.divide()`, and `Tile::at` yields `space.divide()`), but
`Tile::copy_from` is public and the two spaces need not agree there. A mismatch mis-addresses
silently, or panics deep inside `Space::position`. Add a comptime assert in the gathered branch
that the source's projected axes and extents match `space`.

### 2. Width mismatch in the same place

`axis_projection(..., src.store.vector_size)` while every other width in `fill_straight` is
`self.store.vector_size` (`W` / `WP2`). Equal in the staging path, unchecked in general. Use one
consistently, or assert equality.

### 3. The quant assert is unreachable

`Tile::of_impl` already asserts `quant.is_none() || coords.is_direct()`, so a gathered source can
never be quantized. Keeping the assert as defense is fine, but its comment describes a scenario
that cannot exist. Point it at the `of_impl` invariant instead.

### 4. The plane-stage assert is a hard dead end

`crates/cubek-tile/src/staging/fill.rs`.

Gathered plus `Staged` used to be rejected uniformly. It now works for `OperandStage::Smem` and
panics for `OperandStage::Plane`, so a convolution on a cmma leaf has no path at all. Confirm that
is intended, and consider naming the workaround in the message.

## Test gaps

- No staged test combines `in_v = 2` with `checked = true`. That is exactly the interaction the
  change touches: the line-index fold under the projection together with the checked read at the
  window. Both are covered separately, never together, in either schedule.
- `conv2d::check_at` exposes no `checked` knob, so two gathered physical axes plus a mask is
  untested.
- `conv1d_double_buffered` sits under the "staged" heading but is not named or documented as one
  of the staged twins. Cosmetic.

## Simplification

`fill_straight`'s gathered branch is exactly
`src.masked::<WP2, CoordsDyn, AxisProjection>(layout)`: same view stack, same check flag. That
branch is provably plain, since the quant path asserts `!gathered` above it. Delegating drops
about ten lines and keeps the mask flag in one place. Confirm first that `lines::<W>` and
`lines_storage::<T, W>` are identical for a plain store.

## Documentation suggestion

Staging a gathered operand replicates elements: the stage is `tile_oh * rh * ci`, against the
smaller overlapping physical span it reads from. At stride 1 that is roughly an `rh` times shared
memory blow-up over the `Direct` read. That is the reason to keep `Direct` as a real option, and
it deserves a line in `Staging::new`'s doc.
