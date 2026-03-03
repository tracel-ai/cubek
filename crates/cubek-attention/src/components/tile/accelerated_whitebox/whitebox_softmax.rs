use cubecl;
use cubecl::cmma::MmaDefinition;
use cubecl::prelude::*;

use crate::components::tile::accelerated_whitebox::fragment_convert::FragmentConvert;
use crate::components::tile::accelerated_whitebox::manual_matrix::{
    ManualMatrix, ManualMatrixLayout,
};
use crate::components::tile::accelerated_whitebox::setup::WhiteboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::{SoftmaxPipeline, SoftmaxPipelineExpand, SoftmaxRowwise};
use crate::definition::AttentionTileSize;

#[derive(CubeType)]
/// Handles cases where the unit layout is unknown.
///
/// Performs:
/// - storing the score matmul result in shared memory,
/// - loading it into a known layout ([LocalTile]) for computations,
/// - storing back to shared memory (with cast if needed),
/// - loading it in the value LHS format.
pub struct WhiteboxSoftmaxPipeline<
    Acc: Float,
    Lhs: Float,
    FC: FragmentConvert<Acc = Acc, Lhs = Lhs>,
> {
    // Accumulator of score matmul
    pub score_acc: ManualMatrix<Acc>,
    // Lhs of value matmul
    pub value_lhs: ManualMatrix<Lhs>,
    pub transit: FC::Transit,
}

#[cube]
impl<Acc: Float, Lhs: Float, FC: FragmentConvert<Acc = Acc, Lhs = Lhs>>
    WhiteboxSoftmaxPipeline<Acc, Lhs, FC>
{
    pub fn new<Q: Float, K: Float, V: Float, O: Float>(
        transit: FC::Transit,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] _config: WhiteboxAcceleratedAttentionMatmulConfig,
    ) -> Self {
        let score_matmul_tile_size = tile_size.to_score_matmul_tile_size();
        let score_acc = ManualMatrix::<Acc>::new(ManualMatrixLayout::new(
            score_matmul_tile_size,
            cmma::MatrixIdent::Accumulator,
            &MmaDefinition::<Q, K, Acc>::new(
                score_matmul_tile_size.m as usize,
                score_matmul_tile_size.n as usize,
                score_matmul_tile_size.k as usize,
            ),
        ));

        let value_matmul_tile_size = tile_size.to_value_matmul_tile_size();
        let value_lhs = ManualMatrix::<Lhs>::new(ManualMatrixLayout::new(
            value_matmul_tile_size,
            cmma::MatrixIdent::A,
            &MmaDefinition::<Lhs, V, O>::new(
                value_matmul_tile_size.m as usize,
                value_matmul_tile_size.n as usize,
                value_matmul_tile_size.k as usize,
            ),
        ));

        WhiteboxSoftmaxPipeline::<Acc, Lhs, FC> {
            score_acc,
            value_lhs,
            transit,
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float, FC: FragmentConvert<Acc = Acc, Lhs = Lhs>> SoftmaxPipeline<Acc>
    for WhiteboxSoftmaxPipeline<Acc, Lhs, FC>
{
    type ScoreAccFormat = ManualMatrix<Acc>;
    type ValueLhsFormat = ManualMatrix<Lhs>;
    type Rowwise = ManualMatrix<Acc>;
    type Layout = <Self::Rowwise as SoftmaxRowwise<Acc>>::Layout;
    type Transit = FC::Transit;

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        &mut self.score_acc
    }

    fn finalize_lhs(&mut self) {
        FC::acc_to_lhs(&self.score_acc, &mut self.value_lhs, &mut self.transit);
    }

    fn zero(&mut self) {
        self.score_acc.zero()
    }

    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit {
        FC::transit(tile_size, num_planes)
    }
}
