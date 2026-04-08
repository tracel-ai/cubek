use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d},
};
use cubek_std::{stage::StageMemoryConfig, tile::StridedTile};

use crate::components::{
    global::{
        GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
        PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
        read::tiled::{TiledCoords, TiledLayout},
    },
    stage::{StagePartitioner, UnitPartitioner},
};
use crate::definition::{MatrixTypes, StageIdent};

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a unit for each tile
pub struct UnitWriter<IP: MatrixTypes> {
    global: View<Vector<IP::Global, IP::GlobalSize>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<IP::Stage, IP::StageSize>,

    #[cube(comptime)]
    smem_config: StageMemoryConfig,
}

#[cube]
impl<IP: MatrixTypes> UnitWriter<IP> {
    pub fn new(
        global: View<Vector<IP::Global, IP::GlobalSize>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let smem_config = config.smem_config;
        let stage = PartitionedStage::new(
            UnitPartitioner::coordinates(
                config.plane_flow_partition_rule,
                config.plane_dim,
                smem_config.partitions_per_stage_along_col,
            ),
            smem_config,
        );

        UnitWriter::<IP> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, smem_config)),
            stage,
            smem_config,
        }
    }

    fn write(&mut self, tile: Coords2d) {
        unit_write(
            &mut self.global,
            &self.stage.unit_tile,
            tile,
            self.smem_config.comptime().elements_per_tile(),
        )
    }
}

#[cube]
pub fn unit_write<ES: Numeric, NS: Size, EG: Numeric, NG: Size>(
    global: &mut View<Vector<EG, NG>, TiledCoords, ReadWrite>,
    smem_tile: &StridedTile<ES, NS, ReadWrite>,
    tile_pos: Coords2d,
    #[comptime] elements_in_tile: u32,
) {
    let output_vector_size = global.vector_size();
    let out_smem_stage = smem_tile.container.with_vector_size::<NG>();

    let num_vectors = elements_in_tile / output_vector_size as u32;

    for i in 0..num_vectors {
        let value = out_smem_stage[smem_tile.stage_offset(i) as usize];
        global.write_checked(
            (tile_pos, i * output_vector_size as u32),
            Vector::cast_from(value),
        );
    }
}

#[cube]
impl<IP: MatrixTypes> WriteEventListener for UnitWriter<IP> {
    fn on_event(this: &mut Self, event: super::WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => this.write(tile),
            _ => {}
        }
    }
}

#[cube]
impl<IP: MatrixTypes> GlobalWriter<IP> for UnitWriter<IP> {
    type Stage = PartitionedStage<IP::Stage, IP::StageSize>;

    fn init(
        tensor: View<Vector<IP::Global, IP::GlobalSize>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        Self::new(tensor, config)
    }

    fn stage(this: &Self) -> Self::Stage {
        this.stage
    }
}

pub struct UnitWriterFamily;

impl GlobalWriterFamily for UnitWriterFamily {
    type Stage = PartitionedStageFamily;
    type Writer<IP: MatrixTypes> = UnitWriter<IP>;
}
