use crate::{
    components::{
        global::{
            GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
            PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
            read::tiled::{TiledCoords, TiledLayout},
        },
        stage::{PlanePartitioner, StagePartitioner},
    },
    definition::{MatrixTypes, StageIdent},
};
use cubecl::{prelude::*, std::tensor::View, std::tensor::layout::Coords2d};
use cubek_std::{stage::StageMemoryConfig, tile::StridedTile};

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a plane for each tile
pub struct PlaneWriter<IP: MatrixTypes> {
    global: View<Vector<IP::Global, IP::GlobalSize>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<IP::Stage, IP::StageSize>,

    #[cube(comptime)]
    plane_dim: u32,
    #[cube(comptime)]
    smem_config: StageMemoryConfig,
}

#[cube]
impl<IP: MatrixTypes> PlaneWriter<IP> {
    pub fn new(
        global: View<Vector<IP::Global, IP::GlobalSize>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage = PartitionedStage::new(
            PlanePartitioner::coordinates(
                config.plane_flow_partition_rule,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            config.smem_config,
        );

        PlaneWriter::<IP> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            plane_dim: config.plane_dim,
            smem_config: config.smem_config,
        }
    }

    fn write(&mut self, tile_pos: Coords2d) {
        plane_write::<IP::Stage, IP::StageSize, IP::Global, IP::GlobalSize>(
            &mut self.global,
            &self.stage.unit_tile,
            tile_pos,
            self.plane_dim,
            self.smem_config.comptime().elements_per_tile(),
        )
    }
}

#[cube]
impl<IP: MatrixTypes> WriteEventListener for PlaneWriter<IP> {
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
impl<IP: MatrixTypes> GlobalWriter<IP> for PlaneWriter<IP> {
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

#[cube]
pub fn plane_write<ES: Numeric, NS: Size, EG: Numeric, NG: Size>(
    global: &mut View<Vector<EG, NG>, TiledCoords, ReadWrite>,
    smem_tile: &StridedTile<ES, NS, ReadWrite>,
    tile_pos: Coords2d,
    #[comptime] plane_dim: u32,
    #[comptime] elements_in_tile: u32,
) {
    let output_vector_size = global.vector_size().comptime();

    let unit_step = plane_dim * output_vector_size as u32;
    let num_unit_writes = elements_in_tile.div_ceil(unit_step);
    let balanced_workload = elements_in_tile.is_multiple_of(unit_step);

    #[unroll(num_unit_writes == 1)]
    for i in 0..num_unit_writes {
        let unit_write = UNIT_POS_X * output_vector_size as u32 + i * unit_step;

        #[allow(clippy::collapsible_else_if)]
        if balanced_workload {
            write_vector(global, smem_tile, unit_write, tile_pos);
        } else {
            if unit_write < elements_in_tile {
                write_vector(global, smem_tile, unit_write, tile_pos);
            }
        }
    }
}

#[cube]
fn write_vector<ES: Numeric, NS: Size, EG: Numeric, NG: Size>(
    view: &mut View<Vector<EG, NG>, TiledCoords, ReadWrite>,
    out_smem_tile: &StridedTile<ES, NS, ReadWrite>,
    unit_write: u32,
    tile: Coords2d,
) {
    let output_vector_size = view.vector_size().comptime();
    let out_smem_vector_size = out_smem_tile.stage.vector_size().comptime();

    let value = if output_vector_size == out_smem_vector_size {
        let offs = out_smem_tile.stage_offset(unit_write / output_vector_size as u32);
        out_smem_tile.stage[offs as usize]
    } else if out_smem_vector_size < output_vector_size
        && output_vector_size.is_multiple_of(out_smem_vector_size)
    {
        let mut value = Vector::empty();
        #[unroll]
        for i in 0..output_vector_size / out_smem_vector_size {
            let offs = out_smem_tile.stage_offset(unit_write + i as u32);
            #[unroll]
            for j in 0..out_smem_vector_size {
                value[i * out_smem_vector_size + j] = out_smem_tile.stage[offs as usize][j];
            }
        }
        value
    } else {
        unimplemented!()
    };

    view.write_checked((tile, unit_write), Vector::cast_from(value));
}

pub struct PlaneWriterFamily;

impl GlobalWriterFamily for PlaneWriterFamily {
    type Stage = PartitionedStageFamily;
    type Writer<IP: MatrixTypes> = PlaneWriter<IP>;
}
