use cubecl::{
    self as cubecl,
    prelude::*,
    std::tensor::{ViewMut, layout::Coords2d},
};
use cubek_matmul::components::global::{
    GlobalWriterConfig, PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
    plane_write,
    read::tiled::{TiledCoords, TiledLayout},
};
use cubek_std::StageIdent;

use crate::components::{
    global::simple::{AttentionWriter, AttentionWriterExpand},
    stage::{AttentionPartitioner, StageAttentionConfig, plane::PlanePartitioner},
};

#[derive(CubeType)]
pub struct PlaneAttentionWriter<'a, ES: Numeric, ESS: Size, EO: Numeric, EOS: Size> {
    global: ViewMut<'a, Vector<EO, EOS>, TiledCoords>,
    stage: PartitionedStage<ES, ESS>,

    #[cube(comptime)]
    config: GlobalWriterConfig,
}

#[cube]
impl<ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> WriteEventListener
    for PlaneAttentionWriter<'_, ES, ESS, EG, EGS>
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
impl<'a, ES: Numeric, ESS: Size, EG: Numeric, EGS: Size> AttentionWriter<'a, ES, ESS, EG, EGS>
    for PlaneAttentionWriter<'a, ES, ESS, EG, EGS>
{
    fn init<S: StageAttentionConfig>(
        global: ViewMut<'a, Vector<EG, EGS>, Coords2d>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage =
            PartitionedStage::new((PlanePartitioner::seq_q_index(), 0u32), config.smem_config);

        PlaneAttentionWriter::<'a, ES, ESS, EG, EGS> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            config,
        }
    }

    fn stage(&mut self) -> PartitionedStage<ES, ESS> {
        self.stage.clone()
    }
}
