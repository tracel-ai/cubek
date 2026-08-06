# Quantization in the tile API

Where an operand stops being quantized, who decides it, and what is left to build.

## The rule

The space partitioning says how the problem is **cut**. Operands say what they **are**. Formats
never live on the partitioning. Two operands that disagree meet the kind-pairing panics at the
instruction; nothing is cross-validated.

## The model

An operand's quantized form ends at exactly one boundary, stated at launch:

```rust
.leaf(Leaf::Cmma)                                   // what I am at the instruction
.quantized(scales, scheme, Until::Read)             // how far my quantized form travels
```

`Until::Load` — the load into the stage decodes; the stage holds served values, so it inflates by
the served-to-stored ratio and stage depth drops with it.
`Until::Read` — the stage keeps quantized values and their scales; the instruction's read decodes,
amortized over whatever reuse the leaf has.

Which values are available is a **capability**, never a preference. A strided load always decodes;
a TMA bulk copy never does. A memory or manual-mma read decodes; a cmma or `ldmatrix` fragment load
never does. Both facts are the operand's own, so an impossible combination is refused by the
operand builder at the line that states it.

| delivery | leaf | available |
|---|---|---|
| strided | `Memory` / `Mma(Manual)` | `Load` or `Read` — the real knob |
| strided | `Cmma` / `Mma(LoadMatrix)` | `Load` |
| tma | `Memory` / `Mma(Manual)` | `Read` (unwired, see #6) |
| tma | `Cmma` / `Mma(LoadMatrix)` | none — refused |

## Done

1. **Plane-stage guard.** `window_offset` refuses a quantized store, like every sibling door. Was
   silently reinterpreting stored bytes as served values.
2. **`Until` on the operand.** Rides on `QuantInfo`; `MemData::smem_like` reads it and allocates a
   served or stored stage. Deleted `reads_stored`, the `dequant_site` validator, and `quant_pack()`
   at call sites.
3. **Leaf moved off `Partitioner`.** `Partitioner::Final` carries nothing; `Tile.leaf` and
   `TileSpec.leaf` carry it per operand; `Tiling::leaf(..)` became `Tiling::build()`. Deleted
   `Partitioner::leaf()/with_leaf()`, `Space::with_leaf`, `Leaf::is_plane()`, and `mma_leaf`'s
   assert that the two statements agreed.
4. **Manual-mma decodes at the read.** `load_manual` reads through `matrix_transparent`, so a
   packed stage under a plane-level leaf is now possible. `ldmatrix` stays refused.
5. **`Leaf` is pure format.** `{ Memory, Cmma, Mma { io } }`. The `k` came off via
   `promote::<EA, _>(&a)`: a fragment is sized by the whole m×n×k instruction whatever its role, so
   each is missing one dim its own space cannot give. lhs/rhs get theirs from `out`; the
   accumulator now gets `k` from the operand it will contract against.
6. **Quant views ride cubecl-std.** `matrix_transparent`/`flat_transparent` return a plain
   `MatrixView`/`FlatView` in both arms, the quantized one wrapping cubecl's `QuantizedView`.
   Deleted our `QuantizedView`, `unpack_lane`, the `TileView` enum, and four assembly helpers.

## Next

| # | item | notes |
|---|---|---|
| 1 | non-i8 formats (fp8, Q4/Q2 native) | decode is free now (cubecl handles e2m1/e4m3/e5m2); only our storage ladder and `validate_scheme` hardcode i8/u32 |
| 2 | TMA + stored stage | give `TmaData` a scheme; values ride the tensor map, scales ride a plain cooperative copy (they are tiny, so no second descriptor). `size_bytes` is already correct for a quantized destination |
| 3 | requantize on write | ours to build — cubecl has no `QuantizedViewMut`. Line-granular only, which `validate_scheme`'s existing `vector_size % num_quants == 0` already guarantees, so a writer owns whole words and there is no read-modify-write |
| 4 | `MmaIOConfig` narrowing | the three-role bundle can become one `LoadMethod` per operand plus a `StoreMethod` on the accumulator, now that the leaf is per-operand |

## Deferred

- **Scaled MMA (mxfp4/nvfp4).** The instruction eats the format and nothing decodes anywhere;
  turns the impossible tma+cmma cell into the fastest one. cubecl already has `execute_scaled`,
  `new_scaled`, and per-lane `scales_index`. Needs: non-f32 scale params (`QuantParam::UE8M0` /
  `UE4M3`, hard-asserted away in `validate_scheme`), `PackedNative` storage, `QuantLevel::BlockTensor`,
  both-operands-quantized, and scales routed to the *fragment* rather than the view.
- **Computing the scale in-kernel.** Needs a block reduction before writing, so it belongs at drain
  on the accumulator (like softmax's rowwise state), not on a view. Changes what a matmul writes.
- **i8 tensor cores with drain-time scaling.** A different algorithm, not a placement: per-tensor
  and per-N-block scales factor out of the K-contraction, per-K-block scales do not. Largely
  obsoleted by scaled MMA where the hardware has it.

## Known gaps

- `validate_until` runs on both the builder and in `Tile::of_dequant`. The builder is not the only
  door: a `QuantTileArgLaunch` can be built by hand, and test fixtures do exactly that on purpose
  (they *state* an unusual physical layout where the builder *derives* one from a real tensor). So
  the engine validates regardless of the path taken to reach it.

- Scale element is still `f32` (`QuantInfo.buffer: Box<[f32]>`). cubecl's view is generic over it,
  so this is now a field type rather than anything structural.
- `load_manual` refuses a col-major stage: the view is shaped by the stage's space, so a transposed
  read needs a transposing layout, not swapped coordinates. Unreachable today.
- Both operands quantized is refused at the register leaf. Correct there; must not be inherited by
  a scaled-mma rung, where it is the normal case.
