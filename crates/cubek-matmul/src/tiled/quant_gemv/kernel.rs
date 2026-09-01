//! The quantized decode gemv kernel: four operands, one verb.

use cubecl::prelude::*;
use cubek_tile::{Semiring, Space, TileArg};

/// `y = (W ⊗ s) · x`.
///
/// The weight arrives as `u32` words and unpacks at the read ([`TileArg::tile_packed`]); the
/// scales arrive as their own tensor at their own element type and fold in at the contraction
/// ([`Tile::mm_scaled`](cubek_tile::Tile::mm_scaled)). Nothing here mentions a quantization
/// scheme, a block size or a scale binding riding the weight: which values one scale covers is
/// the scales operand's own axes, stated in the space.
///
/// Every operand keeps its own element: `EC` is what the words decode to, `EX` what the
/// activation buffer holds, `ES` the scales', `EO` the output's, and the leaf casts each into
/// the accumulator as it always does. The activation is served in `VX`-wide lines along the
/// contraction — one stored word's worth a step — and the output scalar, because each lane
/// holds a partial of its group's cell.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv_kernel<EC: Numeric, EX: Numeric, ES: Numeric, EO: Numeric, VX: Size, VO: Size>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, EX, VX>,
    scales: &Sequence<TileArg<'_, ES, Const<1>>>,
    out: &TileArg<'_, EO, VO>,
    #[comptime] space: Space,
    #[define(EC)] _served_dtype: ElemType,
    #[define(EX)] _x_dtype: ElemType,
    #[define(ES)] _scale_dtype: ElemType,
    #[define(EO)] _out_dtype: ElemType,
) {
    let w = w.tile_packed::<EC>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let mut scale_tiles = Sequence::new();
    #[unroll]
    for k in 0..scales.len() {
        scale_tiles.push(scales.index(k).tile(comptime!(space.clone())));
    }
    let mut out = out.tile(space);
    out.mm_scaled(&w, &x, &scale_tiles, Semiring::SUM_PROD);
}
