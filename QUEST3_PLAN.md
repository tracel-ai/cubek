# Quest 3: scales explicit, on tensor cores

Every scale says which side and which level it belongs to, is read in its own narrow type inside
the kernel, and the scaled contraction runs on the tensor cores by landing f16 fragments. This
plan continues `QUANT_PLAN.md`; the vocabulary is that file's. Trees: `cubek-Ringo` at
`626548f5` (two commits past metabolic's pin `a40c70f2`), `metabolic-Ringo` at `7c76c45d`.

## What stands

- `mm_scaled(lhs, rhs, &Scales, semiring)` takes one scales operand. `scale_side` reads the side
  off the scales' axes and defaults to the lhs when they share no matrix axis
  (`instruction/registers/contract/scale.rs`). Both consumers, the memory nest
  (`direct.rs::contract_scaled`) and the promoted block (`promoted.rs::mma_scaled`), carry a
  `match side` with two copies of the same body.
- A fragment accumulator refuses a scaled contraction (`ops/matmul/lower.rs`,
  `PlaneTile::mma_scaled`). The fragment fill is `PlanePartition::fill_from`, a `cmma::load`
  from a memory window, which a packed source cannot serve.
- `field_decode` serves integer fields and `e2m1`; `E4M3` and `E5M2` are `Unserved`, and
  nothing spells `ue8m0` as a field at all (`tile/packing.rs`).
- metabolic hand-writes each scale level's axes (`scaled_matmul/operands.rs`), picks the
  launch arm with `PackedSide`, widens `ue4m3`/`ue8m0` scales to `f32` on the host
  (`scaled_matmul/scales.rs`), and sends every quantized problem wider than one row to the
  dequantize path or the gemm table (`matmul/selector.rs::select_quantized`).
- `metabolic-John` carries quest 9's folder move (`matmul/scaled`, two commits, unpushed).
  The metabolic diffs below stay small and semantic so that move can carry them.

## Decisions

1. **Explicit sides, one type.** `Scaling<S> { lhs: ComptimeOption<Scales<S>>, rhs: ComptimeOption<Scales<S>> }`
   with `Scaling::lhs(s)` and `Scaling::rhs(s)`. `mm_scaled`, `mma_scaled` and
   `mma_scaled_with` take `&Scaling<S>`. `ScaleSide` and `scale_side` go; the consumers read the
   side off which option is present. Both present is refused with a message naming quest 4;
   neither present is refused with "use `mm`". `ScalesArg` at launch is unchanged; the kernel
   wraps it. QUANT_PLAN's item 1 (`a.scaled(&s)` through an operand trait) stays the longer
   road; `Scaling` is what it would produce at the leaf, so it is not a detour.
2. **The block level stated at the binding.** `Projection::scales_of(values, inside)` derives a
   scales projection that spans the values' axes and omits `inside`, asserting `inside` is the
   inner digit of a split dim; `Projection::global_of(values)` spans them and addresses none.
   The host says "tied to `KB`" and never writes `[M, KB]`.
3. **Narrow scales through the packed view.** `Packing::Packed { field: Field }` with
   `Field::{Quant(QuantValue), Fp8(Fp8Format)}` and `From<QuantValue>`, so every `.packed(..)`
   call compiles unchanged. `field_decode` serves `Fp8(E4M3 | E5M2 | UE8M0)` through cubecl's
   `fp8_bits_to_f32` and `ue8m0_bits_to_f32`, the byte picked out of its word. A scales buffer
   of bytes binds as `u32` words with `.packed(Field::Fp8(..))` and is served as `f32`. The
   memory nest and the promoted block on the lhs side read one scale a step under a runtime
   ordinal, so the packed view also serves a **sub-word line**: bound width one word, served
   width one value, the byte selected by the position's remainder at runtime. `ue4m3` is `E4M3`
   with the sign bit clear.
4. **The fragment leaf lands the scaled window in a per-plane scratch.** On Metal a cmma
   fragment loads from memory only, so "unpack to f16 fragments in registers" is spelled as: each
   lane writes its `value ⊗ scale` lines, read through the packed view, into a per-plane
   `Shared<[E]>` sized to one fragment window; `sync_plane`; `load_scratch`; the plain
   `cmma::execute`. `mma_leaf_scaled` gains the `PlaneTile::Cmma` and `PlanePartition` arms
   and allocates the scratch once, outside its unrolled fragment loops, with the plane count and
   lane count read off the accumulator's levels. The manual-mma form with register operands
   (Marlin proper) stays refused: it needs NVIDIA to verify and is the speed half of the rung.
5. **metabolic states its scales and drops the widening.** `ScaledOperands` carries per side an
   optional `ScaleSet { block, global, ties_to: Axis }`; the launch derives the projections
   through decision 2. `PackedSide` stays (quest 9's commit 3 deletes it); the kernel spells
   `Scaling::lhs`/`Scaling::rhs` from it. The widen kernel and `needs_widening` go; the
   minifloat bytes bind as words through decision 3.
6. **Prefill routes to the fragment leaf on the N-packed form.** The scaled matmul's derivation
   offers a `Fragments` microkernel ahead of `Registers` where the device offers a cmma shape for
   `(x, x, f32)` and the rows are a multiple of its `m`; the plan gains a `K` walk level and a
   fragment level; the kernel body branches on the comptime microkernel. The selector's heuristic
   arm sends any row count the scaled matmul serves to it. The K-packed form serves one row until
   quest 9's commit 3 puts the weight on the rhs with a straight output; after that the same arm
   serves it, since `Scaling::rhs` and the straight output are what it already assumes.

## Steps, each green on its own

Prefix `C` is `cubek-Ringo` (branch `quest3/explicit-scales`), `M` is `metabolic-Ringo`
(branch `metabolic-ringo`, wired to `../cubek-Ringo` by a LOCAL-ONLY `[patch]` that never ships).

| # | step | proves it |
|---|---|---|
| C0 | Baseline: `cubek-tile` tests on Metal (`tests/tile/{scaled,packed,blocked,decode_gemv}.rs`), note what is red before anything moves | the counts |
| C1 | `Scaling`; `mm_scaled`/`mma_scaled`/`mma_scaled_with` take it; `ScaleSide`, `scale_side` deleted; `ScaleLevel::of` takes the side as a fact; both/neither refused; `cubek-matmul`'s `quant_gemv` kernel and every test kernel spell their side | the four suites unchanged, plus `both_sides_scaled_is_refused_until_quest_4` |
| C2 | `Projection::scales_of` / `global_of`; the scaled and packed test specs rewritten through them | the same suites, the specs shorter |
| C3 | `Field`, `field_decode` for `Fp8`, `unpack_line` for a byte field, the sub-word scalar serve | `e4m3_fields_unpack_on_read`, `ue8m0_scales_are_read_as_bytes` (a q4 contraction against `ue8m0` block scales, host reference), `a_packed_word_serves_one_value` |
| C4 | The fragment leaf: `mma_leaf_scaled` on a cmma accumulator through the per-plane scratch | `a_cmma_accumulator_takes_the_scaled_contraction` on the M2's `8x8x8`, both sides, against the register leaf's numbers |
| M0 | Baseline: `scaled_matmul` unit tests, the Metal correctness suite `gemv_quantized_*`, `plan_time_duplicates`, `cargo xtask validate`; the `[patch]` added and `cargo check --features metal` green | the counts |
| M1 | `ScaleSet` per side on `ScaledOperands`, projections through C2, the kernel spelling `Scaling` by the side | unit tests, `gemv_quantized_*` on Metal |
| M2 | The widen kernel deleted; minifloat scales bound as words through C3 | the `ue4m3` widening test replaced by an end-to-end `ue4m3` scheme case in `gemv_quantized_matches_reference_every_scheme` |
| M3 | The `Fragments` microkernel, the `K` walk level, the kernel's fragment arm, the selector's heuristic route, the hardware check | `a_prefill_row_count_elects_fragments_on_a_cmma_device` (fixtures), `a_device_without_the_instruction_keeps_registers`, and a Metal correctness test at rows 64 on an N-packed q4 weight against the dequantized gemm |
| M4 | `cargo xtask validate`, the Metal suites above; `target/environment/default.db*` deleted since the derivation changed | the gate |

No benchmark is in this plan. The proving-ground number (register leaf against the fragment leaf
at prefill shapes, against Marlin) needs the NVIDIA box and comes after.

## Out of scope, named

- QUANT_PLAN item 1 (the operand trait) and item 3 (the scales on the ring).
- Blackwell's block-scaled instruction, quest 4.
- The K-packed form at more than one row: quest 9's commit 3, then a one-line change here.
- The gemm autotune table racing the fragment arm: a candidate row, after the number exists.
