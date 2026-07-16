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
    #[cube(comptime)]
    gmem_config: GlobalMemoryConfig,
}

#[cube]
impl<'a, AP: AttentionPrecision> QueryReader<'a, AP> {
    pub fn new(
        stage_q_offset: u32,
        query: View<'a, Vector<QG<AP>, QGS<AP>>, Coords2d>,
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        let query = query.slice((stage_q_offset, 0), query.shape());

        QueryReader::<'a, AP> { query, gmem_config }
    }

    pub fn get_tile<P: AttentionPartitioner>(
        &self,
        staged: &mut [Vector<QG<AP>, QGS<AP>>],
        #[comptime] shared_stage: bool,
        tile: Coords2d,
        #[comptime] tile_size: AttentionTileSize,
        #[comptime] partition_seq_q: u32,
        #[comptime] _partition_head_dim: u32,
    ) -> StridedTile<QG<AP>, QGS<AP>> {
        let (row_in_partition, col) = tile;

        let row = row_in_partition + P::seq_q_index() * partition_seq_q;

        let vector_size = self.gmem_config.vector_size.comptime() as u32;
        let vectors_per_row = comptime!(tile_size.head_dim / vector_size);
        let num_vectors = comptime!((tile_size.seq_q * vectors_per_row) as usize);
        let row_base = row * tile_size.seq_q;
        let col_base = col * tile_size.head_dim;

        #[comptime]
        if shared_stage {
            // CMMA/WMMA can load only from global or shared memory. Cooperatively
            // materialize the potentially strided/tail tile into this plane's
            // shared-memory slot; a thread-local Array makes CUDA emit a trap-only
            // kernel ("cannot perform wmma load or store on local memory").
            let num_vectors_runtime = comptime!(num_vectors as u32).runtime();
            let plane_offset = UNIT_POS_Y * num_vectors_runtime;
            let mut index = UNIT_POS_PLANE;
            while index < num_vectors_runtime {
                let r = index / vectors_per_row.runtime();
                let v = index % vectors_per_row.runtime();
                staged[(plane_offset + index) as usize] = self
                    .query
                    .read_checked((row_base + r, col_base + v * vector_size));
                index += PLANE_DIM;
            }
            sync_plane();

            let end = plane_offset + num_vectors_runtime;
            StridedTile::<QG<AP>, QGS<AP>>::new_strided(
                &staged[plane_offset as usize..end as usize],
                0,
                num_vectors_runtime,
                vectors_per_row.runtime(),
                Swizzle::none(),
                self.gmem_config.matrix_layout,
            )
        } else {
            // Register attention may load directly from thread-local storage.
            let mut local = Array::<Vector<QG<AP>, QGS<AP>>>::new(num_vectors);
            #[unroll]
            for r in 0..tile_size.seq_q {
                #[unroll]
                for v in 0..vectors_per_row {
                    local[(r * vectors_per_row + v) as usize] = self
                        .query
                        .read_checked((row_base + r, col_base + v * vector_size));
                }
            }

            StridedTile::<QG<AP>, QGS<AP>>::new_strided(
                local.slice(0, num_vectors),
                0,
                comptime!(num_vectors as u32).runtime(),
                vectors_per_row.runtime(),
                Swizzle::none(),
                self.gmem_config.matrix_layout,
            )
        }
    }
}
