use cubecl;
use cubecl::prelude::*;

use crate::components::tile::accelerated_whitebox::fragment_convert::FragmentConvert;
use crate::components::tile::accelerated_whitebox::manual_matrix::{
    IdentA, IdentCD, ManualMatrix, ManualMatrixLayout,
};
use crate::components::tile::accelerated_whitebox::setup::WhiteboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated_whitebox::{ScoreMma, ValueMma};
use crate::components::tile::{SoftmaxPipeline, SoftmaxPipelineExpand, SoftmaxRowwise};
use crate::definition::{AttentionPrecision, AttentionTileSize};

#[derive(CubeType)]
/// Handles cases where the unit layout is unknown.
///
/// Performs:
/// - storing the score matmul result in shared memory,
/// - loading it into a known layout ([LocalTile]) for computations,
/// - storing back to shared memory (with cast if needed),
/// - loading it in the value LHS format.
pub struct WhiteboxSoftmaxPipeline<AP: AttentionPrecision, FC: FragmentConvert<AP>> {
    // Accumulator of score matmul
    pub softmax_acc: ManualMatrix<IdentCD, ScoreMma<AP>>,
    // Lhs of value matmul
    pub softmax_lhs: ManualMatrix<IdentA, ValueMma<AP>>,
    pub transit: FC::Transit,
}

#[cube]
impl<AP: AttentionPrecision, FC: FragmentConvert<AP>> WhiteboxSoftmaxPipeline<AP, FC> {
    pub fn new<Q: Float, K: Float, V: Float, O: Float>(
        transit: FC::Transit,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] _config: WhiteboxAcceleratedAttentionMatmulConfig,
    ) -> Self {
        let score_matmul_tile_size = tile_size.to_score_matmul_tile_size();
        let acc_layout = ManualMatrixLayout::new(score_matmul_tile_size);
        let softmax_acc = acc_layout.create_matrix();

        let value_matmul_tile_size = tile_size.to_value_matmul_tile_size();
        let lhs_layout = ManualMatrixLayout::new(value_matmul_tile_size);
        let softmax_lhs = lhs_layout.create_matrix();

        WhiteboxSoftmaxPipeline::<AP, FC> {
            softmax_acc,
            softmax_lhs,
            transit,
        }
    }
}

#[cube]
impl<AP: AttentionPrecision, FC: FragmentConvert<AP>> SoftmaxPipeline<AP::SoftmaxAcc>
    for WhiteboxSoftmaxPipeline<AP, FC>
{
    type ScoreAccFormat = ManualMatrix<IdentCD, ScoreMma<AP>>;
    type ValueLhsFormat = ManualMatrix<IdentA, ValueMma<AP>>;
    type Rowwise = ManualMatrix<IdentCD, ScoreMma<AP>>;
    type Layout = <Self::Rowwise as SoftmaxRowwise<AP::SoftmaxAcc>>::Layout;
    type Transit = FC::Transit;

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        &mut self.softmax_acc
    }

    fn finalize_lhs(&mut self) {
        FC::acc_to_lhs(&self.softmax_acc, &mut self.softmax_lhs, &mut self.transit);
    }

    fn zero(&mut self) {
        self.softmax_acc.zero()
    }

    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit {
        FC::transit(tile_size, num_planes)
    }
}
