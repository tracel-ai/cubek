# Quantization in the tile API: the explicit route

Where a quantized operand's pieces are named, and what is left to build.

## The rule

**Scales are an operand. Folding them in is a verb.** A quantized tensor is values *and* scales;
an operand's binding can name one thing, so the other cannot ride it. `QuantTileArg`,
`Quantization`, `DequantAt` and `QuantInfo` are the machinery of smuggling the second one through
the first, and they go. What replaces them is a kernel that says what it reads:

```rust
let w = w.tile(space);            // Tile<u32>, honestly words
let s = s.tile(space);            // Tile<f16>, honestly scales
c.mm_scaled(&w, &x, &s, Semiring::SUM_PROD);
```

Everything follows from that. A scale's element type is the type of the tensor bound, because it
*is* a tensor — which is the whole f16-scales problem, dissolved rather than solved. The decode
site is where the verb is written, so `validate_dequant_at`'s table of legal pairings has nothing
left to police: an illegal pairing has no spelling.

The design record is the 2026-08-20 session's, "Space or Verb":
<https://claude.ai/code/artifact/734748ee-f396-4eb0-9410-04793f17b611>. The rule above is that
record's one line — *the space may state structure and placement; anything needing an operand the
space cannot name has to be a verb in the kernel* — pointed at quantization.

## Landed

1. **The coarse operand needs no engine work** (`tests/tile/coarse.rs`). A scale resolves at a
   coarser level of the same axes, and the plan called that the keystone, to be built as new
   digit-decomposition work. It already exists: a rational projection *is* it.
   `PhysicalAxisMap::of(K).over(block)` is `⌊k / block⌋`, the same floor the resample mapping
   rides. Proven with plain floats at three cuts — equal to the block, finer (several regions read
   one value), coarser (one region addresses several).

   `Tile::copy` refuses a coarse source, and that is the right refusal: a compacted stage fill
   requires source and destination to share a projection. A scale is consumed where the values
   are, never staged into the shape of its own expansion.

2. **`c.mm_scaled(&a, &b, &s, semiring)`** (`ops/matmul/lower.rs`, `tests/tile/scaled.rs`).
   `c = (a ⊗ s) · b`, the scale entering on the product side — one more factor of the term, not
   one more term. The scales ride the walk *beside* the ring rather than in it, so nothing needs a
   three-operand `Staging` payload (the standing open question): one value per block is
   cache-served wherever it sits, and staging it would materialize the expansion the coarse read
   avoids. f16 scales are one of the tests.

   Duplicated bodies rather than an option on the plain path — `contract_scaled` /
   `rank1_update_scaled` (`instruction/registers/block.rs`), `contract_scaled` / `nest_scaled` /
   `body_scaled` (`contract/direct.rs`), `memory_scaled` (`contract/base.rs`). The plain
   contraction is the hot loop of every matmul in the crate, and a scaled step is a different
   program: one more read and one more product per value.

3. **`TileSpec::packed(field)`: packed values without a scheme** (`tests/tile/packed.rs`).
   `Packing` is self-describing now — `Packed { field: QuantValue }`, the field naming its own
   width and sign — and it is *stated* on the operand rather than recovered from a scheme.
   `w.tile_packed::<E>(space)` serves a `u32` binding's fields as values, unpacked at the read by
   `PackedView` (`tile/view/packed.rs`), the unscaled twin of cubecl's `QuantizedView`: no scheme,
   no scale binding, no block grid anywhere in the path.

   `Store` carries the packing, so it is one statement whichever door minted it: a quantized
   operand's scheme derives it (`scheme_packing`), a spec states it, and every "stored is not
   served" refusal keys on it rather than on the presence of scales. A packed source also unpacks
   under `copy_from`, which is decode-into-smem for free when it stages.

   The q4 kernel the plan was blocked on now runs end to end:
   `w.tile_packed()` + a scales tensor + `c.mm_scaled(&w, &x, &s, ..)`. Q4S/Q2S need a device whose
   vectors reach the packing factor (a packed line serves a whole word); verified on cpu, Q8S runs
   everywhere.

4. **The side is read, not stated** (`ScaleSide`, `tests/tile/scaled.rs`, `tests/tile/packed.rs`).
   `mm_scaled` scales whichever operand the scales' *own axes* name: one spanning the output's
   columns is a fact about the rhs's columns and nothing else could fold it in; anything else
   scales the lhs. Same verb, same kernel body, both sides — `(a ⊗ s) · b` and `a · (b ⊗ s)` are
   the same sum of terms, and the side is only where the factor folds in cheapest (once per
   `(row, k)`, or once per `(col, k)` into each rhs line).

   `n` counts the rhs's lines while the scales count their own values, so the rhs read widens the
   column by the accumulator's width; a line still may not straddle a block. A scale over *both*
   matrix axes is refused: that is a scale of the output, not a factor of either term.

5. **The promoted accumulator, and the line rule enforced** (`tests/tile/scaled.rs`).
   `mm_scaled` reaches a register-resident accumulator: `AccumulatorScope::mm_scaled` /
   `mma_scaled`, `PlaneTile::mma_scaled`, `RegisterData::mma_scaled`. The scaled partials no
   longer round-trip through the sink's element between `K` steps, which is the form the decode
   gemv wants. A fragment accumulator still refuses, for the reason under Deferred.

   And the rule the docs stated is now checked: `check_lines_hold_one_scale` reads the block off
   the scales' own projection (`scale_block`) and refuses a step whose line straddles two. Which
   axis the line runs along is the step's own business — `K` past one served value, the
   accumulator's columns at one.

## Next, in order

| # | item | notes |
|---|---|---|
| 3 | **the N-D nest** | `memory_scaled` serves the 2-D nest and refuses the rest loudly. A gathered operand's step has no single scalar `k` to address a scale with; that is a design question, not a copy |
| 4 | **port the quant tests, then delete** | `QuantTileArg`, `Quantization`, `DequantAt`, `validate_dequant_at`, `QuantInfo`'s block bookkeeping, `flat()`'s dequantizing read, `copy_from`'s arithmetic. Acceptance: identical numbers on every existing quant test. **Not a mechanical port** — see the survey below |
| 5 | **the metabolic gemv** | the driver. Its per-token scale-widening pass (~7.9 ms/step of a 75 ms Qwen3-8B decode step) exists only because the engine reads scales at f32; it deletes itself once the gemv is written in this spelling. **Nothing in the engine blocks it any more**: packed values, a scales operand, the rhs side and a promoted accumulator all landed |

### Item 4, surveyed

Three groups, and only the first is a port.

**Ports as it stands** — the register leaf reading its operand in place, one scale level, block or
per-tensor:
`register_matmul_quant_packed_q8` / `_q4`, `register_matmul_quant_rhs_packed_q8` / `_q4`,
`register_matmul_quant_native_block_m`, `register_matmul_quant_native_direct_serve`, the three
`register_matmul_quant_rhs_*_gemv*`, and every `copy_quantized_*` whose point is the *unpack*
rather than the scale. `tests/tile/packed.rs` already carries the lhs and rhs shapes of this in the
new spelling, plus the native one.

**The native store needs nothing at all.** `an_i8_operand_contracts_against_its_scales`: bind the
`i8` tensor as `i8`, `tile()` it, contract. The block casts each value into the accumulator's
element the way it always has, so a store whose element carries no fields never needs a packing
*stated* — a value is whatever its tensor holds, for the same reason a scale is.

**Needs a decision first** (each is a design question, not a copy):

| what | why it does not port |
|---|---|
| every `cmma_matmul_quant_*` | the fragment is loaded from a *staged* operand, so the scale has to fold in at the fill: decode-into-smem, which is Deferred, plus a fill that scales. `mm_scaled` refuses a fragment accumulator on purpose |
| `*_staged_dequantized_smem` | same: a stage holding dequantized values is a *scaled copy*, and `copy_from`'s arithmetic is on the deletion list. So either the verb comes back under its own name, or the stage stays packed and the scale folds at the contraction |
| `copy_quantized_two_level_*` | two scale levels are two scale operands; `mm_scaled` takes one. Either the verb takes a second, or the global scale multiplies into the block scales before the launch (which is what a per-tensor level *is*) |
| `copy_quantized_lookup_*` | a lookup scheme decodes a field through a `2^bits` table. `Packing` names a field's width and sign, not a table; a table is a third operand |
| `copy_quantized_subword_*` | the served line is *narrower* than a stored word. A stated packing ties the served width to `bound_width × factor`, so it cannot spell it. Only the staged fill needs it (`scan_words`) |

The deletions in item 4 are gated on these: `flat()`'s dequantizing read and `copy_from`'s
arithmetic are exactly what the second and third rows still use.

## Deferred

- **Decode into smem** (`dequantize_from` under a hand-written fill). Costs the walk loop, ~8
  lines, and buys nothing until a routine has reuse to amortize it. `mm_scaled` is decode-at-read,
  which is what a decode gemv wants.
- **Scaled MMA (mxfp4/nvfp4).** The instruction eats the format; scales route to the *fragment*,
  not the view. A different rung — and `mma_leaf_scaled` refuses a fragment accumulator today
  precisely so the two do not get confused.
- **The scale factored out of the region.** Where a region sits inside one block, the scale
  multiplies that region's whole contraction once instead of every step — llama.cpp's shape. Valid
  for K-blocks only when the cut divides them, and it changes what a region commits, so it is a
  second contraction shape rather than a tuning knob.
- **Computing the scale in-kernel.** Needs a block reduction before writing, so it belongs at
  drain on the accumulator, not on a view.

## Known gaps

- The straddle check is a comptime assert *inside* the kernel, so it lands on a worker thread and
  the launch returns zeros beside it. Every other contract assert is the same, but a launch-side
  check would read as a rejection; it needs a routine that states the block.
- A packed operand cannot be *staged in its packed form*: `smem_stored` keys the stored stage on
  the scheme, so a stated packing stages unpacked (correct, just larger). Wanted where a routine
  has reuse to amortize; see Deferred.
- `Packing::Native` cannot be *stated* on a spec: the unpacking view is `u32`-only, and the
  refusal says what to do instead (bind the tensor at its own element and let the contraction cast
  it, which is what `an_i8_operand_contracts_against_its_scales` does). It stays reachable through
  a scheme, which is the only thing that mints it.
- The old machinery is untouched and still shipping. Both spellings compile; nothing is deleted
  until item 4.
