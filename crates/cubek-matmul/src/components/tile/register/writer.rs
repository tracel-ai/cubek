use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::tile::register::{UNROLL, UnitFragment, config::RegisterMatmulConfig};

/// Writer for the register matmul fragments.
#[derive(CubeType)]
pub struct RegisterStageWriter {}

#[cube]
impl RegisterStageWriter {
    pub fn store_fragment<A: Numeric, E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        acc: &UnitFragment<A>,
        #[comptime] config: RegisterMatmulConfig,
    ) {
        let out_line_size = tile.stage.line_size().comptime() as u32;

        #[unroll(UNROLL)]
        for i in 0..config.shared.tile_size.mn() / out_line_size {
            let offs = tile.stage_offset(i);
            let mut line = Line::<A, N>::empty();
            #[unroll]
            for j in 0..out_line_size {
                line[j as usize] = acc.array[(i * out_line_size + j) as usize];
            }
            tile.stage[offs as usize] = Line::cast_from(line);
        }
    }
}
