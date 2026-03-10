use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::tile::plane_vec_mat_inner_product::LineContainer;

/// Writer for the output of the VecMat operation.
#[derive(CubeType)]
pub struct MatrixStageWriter {}

#[cube]
impl MatrixStageWriter {
    pub fn store_fragment<A: Numeric, S: Numeric, N: Size>(
        tile: &mut StridedTile<S, N, ReadWrite>,
        acc: &Sequence<LineContainer<A>>,
        #[comptime] n: u32,
        #[comptime] reduce_line_size: VectorSize,
    ) {
        if UNIT_POS_X == 0 {
            let out_line_size = tile.stage.vector_size().comptime();
            let total_out_lines = n as usize / out_line_size;
            #[unroll]
            for out_line_iter in 0..total_out_lines {
                let mut out_line = Vector::<S, N>::empty();

                #[unroll]
                for within_line in 0..out_line_size {
                    let n = out_line_iter * out_line_size + within_line;

                    let line_container = &acc[n];
                    let mut sum = A::from_int(0);
                    for i in 0..reduce_line_size {
                        sum += line_container.line[i];
                    }

                    out_line[within_line] = S::cast_from(sum);
                }

                let offset = tile.stage_offset(out_line_iter as u32);

                tile.stage[offset as usize] = out_line;
            }
        }
    }
}
