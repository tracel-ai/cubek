use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::{
        AccumulatorPipeline, AccumulatorPipelineExpand,
        accelerated::pipeline::whitebox::rowaware_matrix::{
            RowAwareMatrixAccumulator, RowAwareMatrixLayout,
        },
    },
    definition::AttentionTileSize,
};

#[derive(CubeType)]
/// Operates directly on cmma accumulator fragment
pub struct WhiteboxAccumulatorPipeline<E: Float> {
    // Accumulator of value matmul
    pub rowaware_matrix: RowAwareMatrixAccumulator<E>,
}

#[cube]
impl<E: Float> WhiteboxAccumulatorPipeline<E> {
    pub fn new(#[comptime] tile_size: AttentionTileSize) -> Self {
        let fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.seq_q as usize,
                tile_size.val_dim as usize,
                tile_size.seq_kv as usize,
                cmma::MatrixLayout::Undefined,
            )
        };

        let rowaware_matrix = RowAwareMatrixAccumulator::<E> {
            fragment,
            layout: RowAwareMatrixLayout {
                matrix_ident: cmma::MatrixIdent::Accumulator,
            },
        };

        WhiteboxAccumulatorPipeline::<E> { rowaware_matrix }
    }
}

#[cube]
impl<E: Float> AccumulatorPipeline<E> for WhiteboxAccumulatorPipeline<E> {
    type MatmulAccumulator = RowAwareMatrixAccumulator<E>;
    type Rowwise = RowAwareMatrixAccumulator<E>;
    type Transit = ();

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        &mut self.rowaware_matrix
    }

    fn finalize_acc(&mut self) {
        // Nothing to do
    }

    fn zero(&mut self) {
        cmma::fill(&self.rowaware_matrix.fragment, E::from_int(0));
    }

    fn transit(
        #[comptime] _tile_size: AttentionTileSize,
        #[comptime] _num_planes: usize,
    ) -> Self::Transit {
        // Nothing to do
    }
}
