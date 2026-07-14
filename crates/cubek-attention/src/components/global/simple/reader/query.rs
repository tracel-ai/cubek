use cubecl;
use cubecl::{
    prelude::*,
    std::Swizzle,
    std::tensor::{View, layout::Coords2d},
};
use cubek_matmul::components::global::memory::GlobalMemoryConfig;
use cubek_std::tile::StridedTile;

use crate::{
    components::stage::AttentionPartitioner,
    forward::definition::attention_types::{QG, QGS},
    forward::definition::{AttentionPrecision, AttentionTileSize},
};

#[derive(CubeType)]
pub struct QueryReader<'a, AP: AttentionPrecision> {
    query: View<'a, Vector<QG<AP>, QGS<AP>>, Coords2d>,
    stride_row: u32,
    #[cube(comptime)]
    gmem_config: GlobalMemoryConfig,
}

#[cube]
impl<'a, AP: AttentionPrecision> QueryReader<'a, AP> {
    pub fn new(
        stage_q_offset: u32,
        query: View<'a, Vector<QG<AP>, QGS<AP>>, Coords2d>,
        stride_row: u32,
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        let query = query.slice((stage_q_offset, 0), query.shape());

        QueryReader::<'a, AP> {
            query,
            stride_row,
            gmem_config,
        }
    }

    pub fn get_tile<P: AttentionPartitioner>(
        &self,
        tile: Coords2d,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] partition_seq_q: u32,
        #[comptime] _partition_head_dim: u32,
    ) -> StridedTile<QG<AP>, QGS<AP>> {
        let (row_in_partition, col) = tile;

        let row = row_in_partition + P::seq_q_index() * partition_seq_q;

        let vector_size = self.gmem_config.vector_size.comptime() as u32;

        let view = self.query.slice(
            (row * tile_size.seq_q, col * tile_size.head_dim),
            (tile_size.seq_q, tile_size.head_dim).runtime(),
        );
        // The slice is addressed linearly in the underlying storage, so tile
        // rows advance by the tensor's real seq-row stride (permuted query
        // views make it larger than the packed `head_dim`).
        let slice = view.as_linear_slice();

        let start = 0;
        let vectors_per_row = self.stride_row / vector_size;
        let end = start + (tile_size.seq_q - 1) * vectors_per_row
            + tile_size.head_dim / vector_size;

        StridedTile::<QG<AP>, QGS<AP>>::new_strided(
            slice,
            start,
            end,
            vectors_per_row,
            Swizzle::none(),
            self.gmem_config.matrix_layout,
        )
    }
}
