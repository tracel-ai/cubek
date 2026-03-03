use cubecl::prelude::*;
use cubecl::{self, cmma::MmaDefinition};

use crate::{
    components::tile::{
        AccumulatorPipeline, AccumulatorPipelineExpand,
        accelerated_whitebox::manual_matrix::{ManualMatrix, ManualMatrixLayout},
    },
    definition::AttentionTileSize,
};

#[derive(CubeType)]
/// Operates directly on cmma accumulator fragment
pub struct WhiteboxAccumulatorPipeline<E: Float> {
    // Accumulator of value matmul
    pub accumulator: ManualMatrix<E>,
}

#[cube]
impl<E: Float> WhiteboxAccumulatorPipeline<E> {
    pub fn new<SM: Float, V: Float>(#[comptime] tile_size: AttentionTileSize) -> Self {
        let matmul_tile_size = tile_size.to_value_matmul_tile_size();
        let accumulator = ManualMatrix::new(ManualMatrixLayout::new(
            matmul_tile_size,
            cmma::MatrixIdent::Accumulator,
            &MmaDefinition::<SM, V, E>::new(
                matmul_tile_size.m as usize,
                matmul_tile_size.n as usize,
                matmul_tile_size.k as usize,
            ),
        ));

        WhiteboxAccumulatorPipeline::<E> { accumulator }
    }
}

#[cube]
impl<E: Float> AccumulatorPipeline<E> for WhiteboxAccumulatorPipeline<E> {
    type MatmulOperand = ManualMatrix<E>;
    type Rowwise = ManualMatrix<E>;
    type Transit = ();

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        &mut self.accumulator
    }

    fn finalize_acc(&mut self) {
        // Nothing to do
    }

    fn zero(&mut self) {
        self.accumulator.zero();
    }

    fn transit(
        #[comptime] _tile_size: AttentionTileSize,
        #[comptime] _num_planes: usize,
    ) -> Self::Transit {
        // Nothing to do
    }
}
