use super::base::CubeCausalConvMicrokernel;
use super::box_lines::BoxLines;
use super::filters::CausalFilters;
use super::past_taps::PastTaps;
use super::tiles::CausalConvTiles;
use cubecl::tile::Region;
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's box of channel lines of one position of a row: each unit's line convolved over its
/// past and the newest value, plus the bias and then the silu, written to the output at `O`. Where
/// the launch advances the ring, the newest value is written as the ring's next column after the
/// taps are read, by the unit that owns the channels.
#[cube]
pub fn cube_causal_conv_microkernel<E: Float, O: Float, N: Size>(
    cube: &Region,
    tiles: &mut CausalConvTiles,
    #[comptime] microkernel: CubeCausalConvMicrokernel,
) {
    let read_columns = microkernel.taps - 1;
    let past = PastTaps::new(cube, &tiles.past, microkernel.advance, read_columns);
    let filters = CausalFilters::new(&tiles.filters, &tiles.bias, microkernel.taps);
    let lines = BoxLines::new::<N>(cube, microkernel.channel_box());
    let mut output = tiles.output.flat_mut::<O, N>();
    for share in cube {
        let line = lines.line(&share);
        if lines.holds(line) {
            let newest = past.newest::<E, N>(line);
            let convolved = filters.convolved_line::<E, N>(&past, newest, line);
            past.advance::<E, N>(line, newest);
            output.write(line, Vector::<O, N>::cast_from(convolved));
        }
    }
}
