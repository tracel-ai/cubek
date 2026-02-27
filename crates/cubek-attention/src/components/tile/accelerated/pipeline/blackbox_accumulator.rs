use cubecl;
use cubecl::prelude::*;
use cubek_matmul::definition::TileSize;

use crate::components::tile::accelerated::setup::BlackboxAcceleratedAttentionMatmulConfig;
use crate::components::tile::accelerated::{LocalTile, LocalTileLayout};
use crate::components::tile::{AccumulatorPipeline, AccumulatorPipelineExpand};

#[derive(CubeType)]
/// Navigates between cmma fragment (for matmuls) and shared memory (for row wise ops)
pub struct BlackboxAccumulatorPipeline<E: Float> {
    // Accumulator of value matmul
    pub acc_fragment: cmma::Matrix<E>,
    // A slice because knows only the slot for this plane
    smem_slice: SliceMut<E>,
    // Where to perform operations in register
    local_tile: LocalTile<E>,
    #[cube(comptime)]
    stride: u32,
}

#[cube]
impl<E: Float> BlackboxAccumulatorPipeline<E> {
    pub fn new(
        shared_memory: &mut SharedMemory<E>,
        #[comptime] tile_size: TileSize,
        #[comptime] config: BlackboxAcceleratedAttentionMatmulConfig,
    ) -> Self {
        let acc_fragment = unsafe {
            cmma::Matrix::<E>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                tile_size.m as usize,
                tile_size.n as usize,
                tile_size.k as usize,
                cmma::MatrixLayout::Undefined,
            )
        };

        let array_tile_layout = LocalTileLayout::new(
            (tile_size.m, tile_size.n),
            config.shared.plane_dim,
            config.inner_layout,
        );

        let local_tile = LocalTile::new(array_tile_layout);

        let smem_slot_size = tile_size.m * tile_size.n;
        let smem_slice_start = UNIT_POS_Y * smem_slot_size;
        let smem_slice = shared_memory.slice_mut(
            smem_slice_start as usize,
            (smem_slice_start + smem_slot_size) as usize,
        );

        BlackboxAccumulatorPipeline::<E> {
            acc_fragment,
            smem_slice,
            local_tile,
            stride: tile_size.n,
        }
    }
}

#[cube]
impl<E: Float> AccumulatorPipeline<E> for BlackboxAccumulatorPipeline<E> {
    type MatmulAccumulator = cmma::Matrix<E>;
    type Rowwise = LocalTile<E>;

    fn rowwise_mut(&mut self) -> &mut Self::Rowwise {
        cmma::store(
            &mut self.smem_slice,
            &self.acc_fragment,
            self.stride,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        self.local_tile.load_from_slice(&self.smem_slice.to_slice());

        sync_cube();

        &mut self.local_tile
    }

    fn finalize_acc(&mut self) {
        self.local_tile.store_to(&mut self.smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &self.acc_fragment,
            &self.smem_slice.to_slice(),
            self.stride,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn zero(&mut self) {
        cmma::fill(&self.acc_fragment, E::from_int(0));
    }
}
