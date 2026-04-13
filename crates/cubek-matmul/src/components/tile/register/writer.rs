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
        let out_vector_size = tile.container.vector_size().comptime() as u32;

        #[unroll(UNROLL)]
        for i in 0..config.shared.tile_size.mn() / out_vector_size {
            let offs = tile.stage_offset(i);
            let mut vector = Vector::<A, N>::empty();
            #[unroll]
            for j in 0..out_vector_size {
                vector[j as usize] = acc.array[(i * out_vector_size + j) as usize];
            }
            tile.container[offs as usize] = Vector::cast_from(vector);
        }
    }
}
