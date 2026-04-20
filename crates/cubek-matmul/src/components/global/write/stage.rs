use crate::components::{
    stage::{
        ContiguousTilingLayout, RowMajorTilingOrder, Stage, StageFamily, StridedStageMemory,
        TilingLayout,
    },
    tile_matmul::{Scope, Tile},
};
use cubecl::{prelude::*, std::tensor::layout::Coords2d};
use cubek_std::{stage::StageMemoryConfig, tile::StridedTile};

pub type WriteTiling = ContiguousTilingLayout<RowMajorTilingOrder>;

pub struct PartitionedStageFamily;

impl StageFamily<ReadWrite> for PartitionedStageFamily {
    type Stage<ES: Numeric, NS: Size, T: TilingLayout> = PartitionedStage<ES, NS>;
}

#[derive(CubeType, Clone, Copy)]
/// Layoutless stage for current writers. Tile only depends on the unit index, not the out tile.
pub struct PartitionedStage<ES: Numeric, NS: Size> {
    /// Underlying shared memory
    _smem: SharedMemory<Vector<ES, NS>>,
    pub unit_tile: StridedTile<ES, NS, ReadWrite>,
}

#[cube]
impl<ES: Numeric, NS: Size> PartitionedStage<ES, NS> {
    /// Instantiate a new stage memory for the given identifier
    pub fn new(
        unit_pos: Coords2d,
        #[comptime] config: StageMemoryConfig,
    ) -> PartitionedStage<ES, NS> {
        let config = comptime![StageMemoryConfig {
            tiles_per_partition_along_row: 1,
            tiles_per_partition_along_col: 1,
            ..config
        }];

        // Needs to be 16-byte aligned for `stmatrix`
        let inner = StridedStageMemory::<ES, NS, WriteTiling>::new_aligned(16usize, config);

        let tile = inner.get_tile_mut(unit_pos);

        PartitionedStage::<ES, NS> {
            _smem: inner.smem,
            unit_tile: tile,
        }
    }
}

#[cube]
impl<ES: Numeric, NS: Size> Stage<ES, NS, ReadWrite> for PartitionedStage<ES, NS> {
    fn tile<Sc: Scope>(this: &Self, _tile: Coords2d) -> Tile<ES, NS, Sc, ReadWrite> {
        Tile::new_SharedMemory(this.unit_tile)
    }
}
