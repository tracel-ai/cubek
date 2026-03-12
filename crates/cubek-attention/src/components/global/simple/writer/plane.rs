use cubecl::prelude::*;
use cubecl::std::tensor::{View, layout::Coords2d};
use cubecl::{self as cubecl};
use cubek_matmul::components::global::{
    GlobalWriterConfig, PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
    plane_write,
    read::tiled::{TiledCoords, TiledLayout},
};
use cubek_matmul::definition::StageIdent;

use crate::components::{
    global::simple::{AttentionWriter, AttentionWriterExpand},
    stage::{AttentionPartitioner, StageAttentionConfig, plane::PlanePartitioner},
};

#[derive(CubeType)]
pub struct PlaneAttentionWriter<ES: Numeric, ESS: Size, EO: Numeric, EOS: Size> {
    global: View<Vector<EO, EOS>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<ES, ESS>,

    #[cube(comptime)]
    config: GlobalWriterConfig,
}

#[cube]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> PlaneAttentionWriter<ES, ESS, EG, EGS> {}

#[cube]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> WriteEventListener
    for PlaneAttentionWriter<ES, ESS, EG, EGS>
{
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => plane_write::<ES, ESS, EG, EGS>(
                &mut this.global,
                &this.stage.unit_tile,
                tile,
                this.config.plane_dim,
                this.config.comptime().smem_config.elements_per_tile(),
            ),
            _ => {}
        }
    }
}

#[cube]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> AttentionWriter<ES, ESS, EG, EGS>
    for PlaneAttentionWriter<ES, ESS, EG, EGS>
{
    fn init<S: StageAttentionConfig>(
        global: View<Vector<EG, EGS>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage =
            PartitionedStage::new((PlanePartitioner::seq_q_index(), 0u32), config.smem_config);

        PlaneAttentionWriter::<ES, ESS, EG, EGS> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            config,
        }
    }

    fn stage(&mut self) -> PartitionedStage<ES, ESS> {
        self.stage
    }
}
