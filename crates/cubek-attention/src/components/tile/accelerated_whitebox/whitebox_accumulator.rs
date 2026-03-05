use cubecl;
use cubecl::prelude::*;

use crate::components::tile::accelerated_whitebox::manual_matrix::{IdentCD, MmaTypes};
use crate::components::tile::accelerated_whitebox::setup::WhiteboxAcceleratedAttentionMatmulConfig;
use crate::{
    components::tile::{
        AccumulatorPipeline, AccumulatorPipelineExpand,
        accelerated_whitebox::manual_matrix::{ManualMatrix, ManualMatrixLayout},
    },
    definition::AttentionTileSize,
};

#[derive(CubeType)]
/// Operates directly on cmma accumulator fragment
pub struct WhiteboxAccumulatorPipeline<MT: MmaTypes> {
    // Accumulator of value matmul
    pub accumulator: ManualMatrix<IdentCD, MT>,
}

#[cube]
impl<MT: MmaTypes> WhiteboxAccumulatorPipeline<MT> {
    pub fn new<SM: Float, V: Float>(
        #[comptime] config: WhiteboxAcceleratedAttentionMatmulConfig,
    ) -> Self {
        let matmul_tile_size = config
            .shared
            .attention_tile_size
            .to_value_matmul_tile_size();
        let layout = ManualMatrixLayout::new(matmul_tile_size, config.value_mma_io_config);
        let accumulator = layout.create_matrix();
        WhiteboxAccumulatorPipeline::<MT> { accumulator }
    }
}

#[cube]
impl<MT: MmaTypes<CD: Float>> AccumulatorPipeline<MT::CD> for WhiteboxAccumulatorPipeline<MT> {
    type ValueAccFormat = ManualMatrix<IdentCD, MT>;
    type Rowwise = ManualMatrix<IdentCD, MT>;
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
