use cubecl;
use cubecl::prelude::*;

use crate::components::tile::accelerated_blackbox::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated_blackbox::{LocalTile, LocalTileLayout};
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
pub struct BlackboxSoftmaxPipeline<Acc: Float, Lhs: Float> {
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
impl<Acc: Float, Lhs: Float> BlackboxSoftmaxPipeline<Acc, Lhs> {
    pub fn new(
        acc_shared_memory: &mut SharedMemory<Acc>,
        lhs_shared_memory: &mut SharedMemory<Lhs>,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] config: BlackboxAcceleratedAttentionMatmulConfig,
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

        BlackboxSoftmaxPipeline::<Acc, Lhs> {
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
impl<Acc: Float, Lhs: Float> SoftmaxPipeline<Acc> for BlackboxSoftmaxPipeline<Acc, Lhs> {
    type ScoreAccFormat = cmma::Matrix<Acc>;
    type ValueLhsFormat = cmma::Matrix<Lhs>;
    type Rowwise = LocalTile<Acc>;
    type Layout = <Self::Rowwise as SoftmaxRowwise<Acc>>::Layout;
    type Transit = (SharedMemory<Acc>, SharedMemory<Lhs>);

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        cmma::store(
            &mut self.acc_smem_slice,
            &self.acc_fragment,
            self.stride,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        self.local_tile
            .load_from_slice(&self.acc_smem_slice.to_slice());

        sync_cube();

        &mut self.local_tile
    }

    fn finalize_lhs(&mut self) {
        self.local_tile.store_to(&mut self.lhs_smem_slice);

        sync_cube();

        cmma::load(
            &self.lhs_fragment,
            &self.lhs_smem_slice.to_slice(),
            self.stride,
        );
    }

    fn zero(&mut self) {
        cmma::fill(&self.acc_fragment, Acc::from_int(0));
    }

    fn transit(
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] num_planes: usize,
    ) -> Self::Transit {
        let smem_size = tile_size.seq_q as usize * tile_size.seq_kv as usize * num_planes;
        (SharedMemory::new(smem_size), SharedMemory::new(smem_size))
    }
}
