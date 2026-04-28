use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d},
    {self as cubecl},
};
use cubek_matmul::components::global::{
    GlobalWriterConfig, PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
    read::tiled::{TiledCoords, TiledLayout},
    unit_write,
};
use cubek_std::StageIdent;

use crate::components::{
    global::simple::{AttentionWriter, AttentionWriterExpand},
    stage::{AttentionPartitioner, StageAttentionConfig, unit::UnitPartitioner},
};

#[derive(CubeType)]
pub struct UnitAttentionWriter<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> {
    global: View<Vector<EG, EGS>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<ES, ESS>,

    #[cube(comptime)]
    config: GlobalWriterConfig,
}

#[cube]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> WriteEventListener
    for UnitAttentionWriter<ES, ESS, EG, EGS>
{
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => unit_write::<ES, ESS, EG, EGS>(
                &mut this.global,
                &this.stage.unit_tile,
                tile,
                this.config.comptime().smem_config.elements_per_tile(),
            ),
            _ => {}
        }
    }
}

#[cube]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> AttentionWriter<ES, ESS, EG, EGS>
    for UnitAttentionWriter<ES, ESS, EG, EGS>
{
    fn init<S: StageAttentionConfig>(
        global: View<Vector<EG, EGS>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage =
            PartitionedStage::new((UnitPartitioner::seq_q_index(), 0u32), config.smem_config);

        UnitAttentionWriter::<ES, ESS, EG, EGS> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            config,
        }
    }

    fn stage(&mut self) -> PartitionedStage<ES, ESS> {
        self.stage
    }
}
