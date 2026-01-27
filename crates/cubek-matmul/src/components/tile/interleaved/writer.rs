use cubecl::prelude::*;

use crate::components::tile::{
    StridedTile,
    interleaved::{InterleavedAccumulator, config::InterleavedMatmulConfig},
};

/// Writer for the interleaved matmul fragments.
///
/// Before writing, sums all the unit accumulators
#[derive(CubeType)]
pub struct InterleavedStageWriter {}

#[cube]
impl InterleavedStageWriter {
    pub fn store_fragment<A: Numeric, E: Numeric>(
        tile: &mut StridedTile<E, ReadWrite>,
        acc: &InterleavedAccumulator<A>,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
        if UNIT_POS_X == 0 {
            let out_line_size = tile.stage.line_size().comptime() as u32;

            #[unroll]
            for i in 0..config.shared.tile_size.mn() / out_line_size {
                let offs = tile.stage_offset(i);
                let mut line = Line::empty(out_line_size as usize);
                #[unroll]
                for j in 0..out_line_size {
                    line[j as usize] = acc.array[(i * out_line_size + j) as usize];
                }
                tile.stage[offs as usize] = Line::cast_from(line);
            }
        }
    }
}
