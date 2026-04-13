use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::tile::interleaved::{
    InterleavedAccumulator, config::InterleavedMatmulConfig,
};

/// Writer for the interleaved matmul fragments.
///
/// Before writing, sums all the unit accumulators
#[derive(CubeType)]
pub struct InterleavedStageWriter {}

#[cube]
impl InterleavedStageWriter {
    pub fn store_fragment<A: Numeric, E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        acc: &InterleavedAccumulator<A>,
        #[comptime] config: InterleavedMatmulConfig,
    ) {
        if UNIT_POS_X == 0 {
            let out_vector_size = tile.container.vector_size().comptime() as u32;

            #[unroll]
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
}
