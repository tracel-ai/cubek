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

        // Stage the tile in registers through checked view reads: the view's
        // layout resolves the tensor's real strides (query views are often
        // permuted — `(b, seq, heads, hd)` swapped to `(b, heads, seq, hd)`),
        // and rows past `seq_q` read zero instead of out of bounds. The
        // q-stage span rarely divides `seq_q`, so tail tiles overhang the
        // tensor — and the memory behind it may end right there (exact-sized
        // persistent allocations do). When `seq_q` divides the span the
        // bounds check is compiled out.
        let mut staged = Array::<Vector<QG<AP>, QGS<AP>>>::new(num_vectors);
        #[unroll]
        for r in 0..tile_size.seq_q {
            #[unroll]
            for v in 0..vectors_per_row {
                staged[(r * vectors_per_row + v) as usize] = self
                    .query
                    .read_checked((row_base + r, col_base + v * vector_size));
            }
        }

        StridedTile::<QG<AP>, QGS<AP>>::new_strided(
            staged.slice(0, num_vectors),
            0,
            comptime!(num_vectors as u32).runtime(),
            vectors_per_row.runtime(),
            Swizzle::none(),
            self.gmem_config.matrix_layout,
        )
    }
}
