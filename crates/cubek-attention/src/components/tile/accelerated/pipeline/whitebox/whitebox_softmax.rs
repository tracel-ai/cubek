use cubecl;
use cubecl::prelude::*;

use crate::components::tile::accelerated::setup::AcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated::{LocalTile, LocalTileLayout};
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
pub struct WhiteboxSoftmaxPipeline<Acc: Float, Lhs: Float> {
    // Accumulator of score matmul
    pub acc_fragment: cmma::Matrix<Acc>,
    // Lhs of value matmul
    pub lhs_fragment: cmma::Matrix<Lhs>,
    acc_smem_slice: SliceMut<Acc>,
    lhs_smem_slice: SliceMut<Lhs>,
    // Where to perform operations in register
    local_tile: LocalTile<Acc>,
    #[cube(comptime)]
    stride: u32,
}

#[cube]
impl<Acc: Float, Lhs: Float> WhiteboxSoftmaxPipeline<Acc, Lhs> {
    pub fn new(
        acc_shared_memory: &mut SharedMemory<Acc>,
        lhs_shared_memory: &mut SharedMemory<Lhs>,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] config: AcceleratedAttentionMatmulConfig,
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

        let lhs_fragment = unsafe {
            cmma::Matrix::<Lhs>::uninitialized(
                cmma::MatrixIdent::A,
                tile_size.seq_q as usize,
                tile_size.val_dim as usize,
                tile_size.seq_kv as usize,
                cmma::MatrixLayout::RowMajor,
            )
        };

        let array_tile_layout = LocalTileLayout::new(
            (tile_size.seq_q, tile_size.seq_kv),
            config.shared.plane_dim,
            config.inner_layout,
        );

        let local_tile = LocalTile::new(array_tile_layout);

        let smem_slot_size = (tile_size.seq_q * tile_size.seq_kv) as usize;
        let smem_slice_start = UNIT_POS_Y as usize * smem_slot_size;
        let smem_slice_end = smem_slice_start + smem_slot_size;

        let acc_smem_slice = acc_shared_memory.slice_mut(smem_slice_start, smem_slice_end);
        let lhs_smem_slice = lhs_shared_memory.slice_mut(smem_slice_start, smem_slice_end);

        WhiteboxSoftmaxPipeline::<Acc, Lhs> {
            acc_fragment,
            lhs_fragment,
            acc_smem_slice,
            lhs_smem_slice,
            local_tile,
            stride: tile_size.seq_kv,
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> SoftmaxPipeline<Acc> for WhiteboxSoftmaxPipeline<Acc, Lhs> {
    type MatmulAccumulator = cmma::Matrix<Acc>;
    type MatmulLhs = cmma::Matrix<Lhs>;
    type Rowwise = cmma::Matrix<Acc>;
    type SoftmaxLayout = <Self::Rowwise as SoftmaxRowwise<Acc>>::Layout;

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        todo!()
    }

    fn finalize_lhs(&mut self) {
        todo!()
    }

    fn zero(&mut self) {
        todo!()
    }
}
