use cubecl;
use cubecl::prelude::*;

use crate::components::tile::accelerated::pipeline::whitebox::fragment_convert::FragmentConvert;
use crate::components::tile::accelerated::pipeline::whitebox::rowaware_matrix::{
    RowAwareMatrix, RowAwareMatrixLayout,
};
use crate::components::tile::accelerated::setup::AcceleratedAttentionMatmulConfig;
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
    pub rowaware_acc: RowAwareMatrix<Acc>,
    // Lhs of value matmul
    pub lhs_fragment: cmma::Matrix<Lhs>,
    pub transit: FC::Transit,
}

#[cube]
impl<Acc: Float, Lhs: Float, FC: FragmentConvert<Acc = Acc, Lhs = Lhs>>
    WhiteboxSoftmaxPipeline<Acc, Lhs, FC>
{
    pub fn new(
        transit: FC::Transit,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] _config: AcceleratedAttentionMatmulConfig,
    ) -> Self {
        let acc_fragment = unsafe {
            cmma::Matrix::<Acc>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.seq_q as usize,
                tile_size.seq_kv as usize,
                tile_size.head_dim as usize,
                cmma::MatrixLayout::Undefined,
            )
        };

        let rowaware_acc = RowAwareMatrix::<Acc> {
            fragment: acc_fragment,
            layout: RowAwareMatrixLayout {},
        };

        let lhs_fragment = unsafe {
            cmma::Matrix::<Lhs>::uninitialized(
                cmma::MatrixIdent::A,
                tile_size.seq_q as usize,
                tile_size.val_dim as usize,
                tile_size.seq_kv as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };

        WhiteboxSoftmaxPipeline::<Acc, Lhs, FC> {
            rowaware_acc,
            lhs_fragment,
            transit,
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float, FC: FragmentConvert<Acc = Acc, Lhs = Lhs>> SoftmaxPipeline<Acc>
    for WhiteboxSoftmaxPipeline<Acc, Lhs, FC>
{
    type MatmulAccumulator = cmma::Matrix<Acc>;
    type MatmulLhs = cmma::Matrix<Lhs>;
    type Rowwise = RowAwareMatrix<Acc>;
    type Layout = <Self::Rowwise as SoftmaxRowwise<Acc>>::Layout;
    type Transit = FC::Transit;

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        &mut self.rowaware_acc
    }

    fn finalize_lhs(&mut self) {
        FC::acc_to_lhs(
            &self.rowaware_acc.fragment,
            &mut self.lhs_fragment,
            &mut self.transit,
        );
    }

    fn zero(&mut self) {
        cmma::fill(&self.rowaware_acc.fragment, Acc::from_int(0));
    }

    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit {
        FC::transit(tile_size, num_planes)
    }
}
