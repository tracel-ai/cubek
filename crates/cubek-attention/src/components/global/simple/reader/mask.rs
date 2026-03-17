use crate::definition::attention_types::{MSK, MSKS};
use crate::definition::{AttentionPrecision, AttentionTileSize};
use cubecl;
use cubecl::prelude::*;
use cubecl::std::tensor::{View, layout::Coords2d};
use cubecl::std::{Swizzle, tensor::layout::Coordinates};
use cubek_matmul::components::global::memory::{GlobalIterator, GlobalMemoryConfig};
use cubek_std::tile::StridedTile;

use crate::components::stage::{AttentionPartitioner, StageAttentionConfig};

#[derive(CubeType)]
pub struct LogicalIterator {
    row: u32,
    col: RuntimeCell<u32>,
    step_col: u32,
}

#[cube]
impl LogicalIterator {
    fn init(stage_q_offset: u32, step_col: u32) -> LogicalIterator {
        LogicalIterator {
            row: stage_q_offset,
            col: RuntimeCell::new(0),
            step_col,
        }
    }

    fn read(&self) -> Coords2d {
        (self.row, self.col.read())
    }

    fn advance(&mut self) {
        self.col.store(self.col.read() + self.step_col);
    }
}

#[derive(CubeType)]
pub struct MaterializedMaskReader<M: Numeric, N: Size> {
    global_iter: GlobalIterator<Vector<M, N>>,
    logical_iter: LogicalIterator,
    // TODO not sure if mandatory, but i need for the stride when reading in global memory
    seq_kv_shape: u32,
    #[cube(comptime)]
    gmem_config: GlobalMemoryConfig,
}

#[derive(CubeType)]
pub enum MaskReader<AP: AttentionPrecision> {
    Materialized(MaterializedMaskReader<MSK<AP>, MSKS<AP>>),
    Logical(LogicalIterator),
}

#[cube]
impl<AP: AttentionPrecision> MaskReader<AP> {
    pub fn new_logical(partition_q_offset: u32, step: u32) -> Self {
        MaskReader::<AP>::new_Logical(LogicalIterator::init(partition_q_offset, step))
    }

    pub fn new_materialized(
        stage_q_offset: u32,
        partition_q_offset: u32,
        mask: View<Vector<MSK<AP>, MSKS<AP>>, Coords2d>,
        step: u32,
        seq_kv_shape: u32,
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        let mask = mask.slice((stage_q_offset, 0), mask.shape());
        let global_iter = GlobalIterator::new(mask, step, gmem_config.view_direction, false);

        MaskReader::<AP>::new_Materialized(MaterializedMaskReader::new(
            global_iter,
            LogicalIterator::init(partition_q_offset, step),
            seq_kv_shape,
            gmem_config,
        ))
    }

    pub fn read<P: AttentionPartitioner, S: StageAttentionConfig>(
        &self,
        #[comptime] pos_in_partition: Coords2d,
        #[comptime] config: S,
    ) -> (Coords2d, ComptimeOption<StridedTile<MSK<AP>, MSKS<AP>>>) {
        let tile_size = config.tile_size();

        let partition_tile_offset = (
            pos_in_partition.0 * tile_size.seq_q,
            pos_in_partition.1 * tile_size.seq_kv,
        );

        let (origin, tile) = match self {
            MaskReader::Materialized(materialized_mask_reader) => (
                materialized_mask_reader.logical_iter.read(),
                ComptimeOption::new_Some(materialized_mask_reader.read::<P>(
                    partition_tile_offset,
                    config.tile_size(),
                    config.elements_in_partition_seq_q(),
                )),
            ),
            MaskReader::Logical(logical_iter) => (logical_iter.read(), ComptimeOption::new_None()),
        };

        (Coords2d::add(origin, partition_tile_offset.runtime()), tile)
    }

    pub fn advance_view(&mut self) {
        match self {
            MaskReader::Logical(logical_iter) => logical_iter.advance(),
            MaskReader::Materialized(materialized_mask_reader) => {
                materialized_mask_reader.advance()
            }
        }
    }
}

#[cube]
impl<M: Numeric, N: Size> MaterializedMaskReader<M, N> {
    fn new(
        global_iter: GlobalIterator<Vector<M, N>>,
        logical_iter: LogicalIterator,
        seq_kv_shape: u32,
        #[comptime] gmem_config: GlobalMemoryConfig,
    ) -> Self {
        MaterializedMaskReader::<M, N> {
            global_iter,
            logical_iter,
            seq_kv_shape,
            gmem_config,
        }
    }

    fn read<P: AttentionPartitioner>(
        &self,
        #[comptime] partition_tile_offset: Coords2d,
        #[comptime] attention_tile_size: AttentionTileSize,
        #[comptime] elements_in_partition_seq_q: u32,
    ) -> StridedTile<M, N> {
        let (row_offset, col) = partition_tile_offset;

        let row = row_offset + P::seq_q_index() * elements_in_partition_seq_q;

        let slice = self
            .global_iter
            .view()
            .slice(
                (row, col.runtime()),
                (attention_tile_size.seq_q, attention_tile_size.seq_kv).runtime(),
            )
            .to_linear_slice();

        let vector_size = self.gmem_config.vector_size.comptime() as u32;
        let start = 0;
        let length = attention_tile_size.seq_q * attention_tile_size.seq_kv / vector_size;
        let end = start + length;
        let stride = self.seq_kv_shape / vector_size;

        StridedTile::<M, N>::new_strided(
            slice,
            start,
            end,
            stride,
            Swizzle::none(),
            self.gmem_config.matrix_layout,
        )
    }

    fn advance(&mut self) {
        self.global_iter.advance();
        self.logical_iter.advance()
    }
}
