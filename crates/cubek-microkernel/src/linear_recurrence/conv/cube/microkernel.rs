use super::base::CubeCausalConvMicrokernel;
use super::filters::CausalConvFilters;
use super::past_reader::PastReader;
use super::run_lines::RunLines;
use super::tiles::CausalConvTiles;
use cubecl::tile::Region;
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's run of channel lines of one position of a row: each unit's line convolved over its
/// past and the newest value, plus the bias and then the silu, written to the output at `O`. Where
/// the launch writes the ring, the newest value is written as the ring's next column after the
/// taps are read, by the unit that owns the channels.
#[cube]
pub fn cube_causal_conv_microkernel<E: Float, O: Float, N: Size>(
    cube: &Region,
    tiles: &mut CausalConvTiles,
    #[comptime] microkernel: CubeCausalConvMicrokernel,
) {
    let read_columns = microkernel.taps - 1;
    let past = PastReader::new(cube, &tiles.past, microkernel.writer, read_columns);
    let filters = CausalConvFilters::new(&tiles.filters, &tiles.bias, microkernel.taps);
    let lines = RunLines::new::<N>(cube, microkernel.channel_run());
    let mut output = tiles.output.flat_mut::<O, N>();
    for share in cube {
        let line = lines.line(&share);
        if lines.holds(line) {
            let newest = past.newest::<E, N>(line);
            let convolved = filters.convolved_line::<E, N>(&past, newest, line);
            past.write_newest::<E, N>(line, newest);
            output.write(line, Vector::<O, N>::cast_from(convolved));
        }
    }
}
