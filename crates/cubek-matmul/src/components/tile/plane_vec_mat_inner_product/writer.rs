use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::tile::plane_vec_mat_inner_product::VectorContainer;

/// Writer for the output of the VecMat operation.
#[derive(CubeType)]
pub struct MatrixStageWriter {}

#[cube]
impl MatrixStageWriter {
    pub fn store_fragment<A: Numeric, S: Numeric, N: Size>(
        tile: &mut StridedTile<S, N, ReadWrite>,
        acc: &Sequence<VectorContainer<A>>,
        #[comptime] n: u32,
        #[comptime] reduce_vector_size: VectorSize,
    ) {
        if UNIT_POS_X == 0 {
            let out_vector_size = tile.container.vector_size().comptime();
            let total_out_vectors = n as usize / out_vector_size;
            #[unroll]
            for out_vector_iter in 0..total_out_vectors {
                let mut out_vector = Vector::<S, N>::empty();

                #[unroll]
                for within_vector in 0..out_vector_size {
                    let n = out_vector_iter * out_vector_size + within_vector;

                    let vector_container = &acc[n];
                    let mut sum = A::from_int(0);
                    for i in 0..reduce_vector_size {
                        sum += vector_container.vector[i];
                    }

                    out_vector[within_vector] = S::cast_from(sum);
                }

                let offset = tile.stage_offset(out_vector_iter as u32);

                tile.container[offset as usize] = out_vector;
            }
        }
    }
}
