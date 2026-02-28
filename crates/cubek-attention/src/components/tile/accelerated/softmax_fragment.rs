use cubecl;
use cubecl::prelude::*;

use crate::components::tile::RowWise;
use crate::components::tile::accelerated::local_tile::{LocalTile, LocalTileLayout};
use crate::components::tile::accelerated::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::{FragmentAccumulator, FragmentAccumulatorExpand};
use crate::components::tile::{FragmentSoftmax, FragmentSoftmaxExpand};
use crate::definition::AttentionTileSize;

#[derive(CubeType)]
/// Navigates between cmma fragment (for matmuls) and shared memory (for row wise ops)
pub struct SoftmaxHybridFragment<Acc: Float, Lhs: Float> {
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
impl<Acc: Float, Lhs: Float> SoftmaxHybridFragment<Acc, Lhs> {
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

        SoftmaxHybridFragment::<Acc, Lhs> {
            acc_fragment,
            lhs_fragment,
            acc_smem_slice,
            lhs_smem_slice,
            local_tile,
            stride: tile_size.seq_kv,
        }
    }

    fn zero(&mut self) {
        cmma::fill(&self.acc_fragment, Acc::from_int(0));
        cmma::fill(&self.lhs_fragment, Lhs::from_int(0));
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> FragmentSoftmax<Acc> for SoftmaxHybridFragment<Acc, Lhs> {
    type Layout = LocalTileLayout;
    type SoftmaxRowFormat = LocalTile<Acc>;

    fn rowwise_mut(&mut self) -> &mut Self::SoftmaxRowFormat {
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

    fn update_from_rowwise(&mut self) {
        self.local_tile.store_to(&mut self.lhs_smem_slice);

        sync_cube();

        cmma::load(
            &self.lhs_fragment,
            &self.lhs_smem_slice.to_slice(),
            self.stride,
        );
    }

    fn zero(&mut self) {
        self.zero();
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> FragmentAccumulator<Acc> for SoftmaxHybridFragment<Acc, Lhs> {
    fn rowwise_scale(&mut self, val: &RowWise<Acc>) {
        let local_tile = self.rowwise_mut();
        local_tile.rowwise_scale(val);
        self.update_from_rowwise();
    }

    fn zero(&mut self) {
        self.zero();
    }
}
