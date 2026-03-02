use cubecl;
use cubecl::prelude::*;
use cubek_matmul::definition::TileSize;

use crate::components::tile::{AccumulatorPipeline, AccumulatorPipelineExpand};

#[derive(CubeType)]
/// Operates directly on cmma accumulator fragment
pub struct WhiteboxAccumulatorPipeline<E: Float> {
    // Accumulator of value matmul
    pub acc_fragment: cmma::Matrix<E>,
}

#[cube]
impl<E: Float> WhiteboxAccumulatorPipeline<E> {
    pub fn new(#[comptime] tile_size: TileSize) -> Self {
        let acc_fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.m as usize,
                tile_size.n as usize,
                tile_size.k as usize,
                cmma::MatrixLayout::Undefined,
            )
        };

        WhiteboxAccumulatorPipeline::<E> { acc_fragment }
    }
}

#[cube]
impl<E: Float> AccumulatorPipeline<E> for WhiteboxAccumulatorPipeline<E> {
    type MatmulAccumulator = cmma::Matrix<E>;
    type Rowwise = cmma::Matrix<E>;

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        &mut self.acc_fragment
    }

    fn finalize_acc(&mut self) {
        // Nothing to do
    }

    fn zero(&mut self) {
        cmma::fill(&self.acc_fragment, E::from_int(0));
    }
}
