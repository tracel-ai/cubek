use crate::multi_level::{
    components::{
        global::{
            GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
            PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
        },
        stage::{PlanePartitioner, partition_coordinates},
    },
    definition::MatrixTypes,
    stage::{StageMemoryConfig, SwizzleMode},
};
use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::Coords2d},
};
use cubek_std::{InvalidConfigError, MatrixLayout};

/// TMA writes shared memory at 128-byte boundaries.
const TMA_ALIGNMENT: usize = 128;

#[derive(CubeType)]
/// Stores each tile from out shared memory to output global memory with one TMA store per
/// plane. The output is a tensor map whose box is one tile.
pub struct TmaWriter<'a, IP: MatrixTypes> {
    global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
    stage: PartitionedStage<IP::Stage, IP::StageSize>,

    #[cube(comptime)]
    smem_config: StageMemoryConfig,
}

#[cube]
impl<'a, IP: MatrixTypes> TmaWriter<'a, IP> {
    pub fn new(
        global: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage = PartitionedStage::new_aligned(
            partition_coordinates::<PlanePartitioner>(
                config.plane_flow_partition_rule,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            TMA_ALIGNMENT,
            config.smem_config,
        );

        TmaWriter::<'a, IP> {
            global,
            stage,
            smem_config: config.smem_config,
        }
    }

    fn write(&mut self, tile: Coords2d) {
        let (rows, cols) = comptime![(
            self.smem_config.elements_per_tile_along_row,
            self.smem_config.elements_per_tile_along_col
        )];
        let smem_tile = &self.stage.unit_tile;
        let source = &smem_tile.container[smem_tile.start as usize..smem_tile.end as usize];

        // The units stored the tile through the generic proxy, and TMA reads it through the
        // async proxy.
        sync_async_proxy_shared();
        sync_plane();
        if UNIT_POS_X == 0 {
            let pos = (tile.0 * rows, tile.1 * cols);
            // The next tile is stored into the same memory.
            self.global.tensor_map_store(source.downcast(), pos).wait();
        }
        sync_plane();
    }
}

#[cube]
impl<IP: MatrixTypes> WriteEventListener for TmaWriter<'_, IP> {
    fn on_event(this: &mut Self, event: super::WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => {
                this.write(tile);
            }
            _ => {}
        }
    }
}

#[cube]
impl<'a, IP: MatrixTypes> GlobalWriter<'a, IP> for TmaWriter<'a, IP> {
    type Stage = PartitionedStage<IP::Stage, IP::StageSize>;

    fn init(
        tensor: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        Self::new(tensor, config)
    }

    fn stage(this: &Self) -> Self::Stage {
        this.stage.clone()
    }
}

pub struct TmaWriterFamily;

impl GlobalWriterFamily for TmaWriterFamily {
    type Stage = PartitionedStageFamily;
    type Writer<'a, IP: MatrixTypes> = TmaWriter<'a, IP>;

    fn validate_with_config(config: &GlobalWriterConfig) -> Result<(), InvalidConfigError> {
        let smem = &config.smem_config;
        // TMA copies bytes: the tile in shared memory is the tile in global memory.
        if smem.dtype != config.gmem_config.dtype {
            return Err(Box::new(format!(
                "TMA writer needs the out stage in the output type, got {:?} for {:?}",
                smem.dtype, config.gmem_config.dtype
            )));
        }
        if smem.matrix_layout != MatrixLayout::RowMajor || smem.swizzle != SwizzleMode::None {
            return Err(Box::new(
                "TMA writer needs an unswizzled row-major out stage".to_string(),
            ));
        }
        let row_bytes = smem.elements_per_tile_along_col as usize * smem.dtype.size();
        if !row_bytes.is_multiple_of(16) {
            return Err(Box::new(format!(
                "TMA writer needs tile rows of a multiple of 16 bytes, got {row_bytes}"
            )));
        }
        Ok(())
    }
}
