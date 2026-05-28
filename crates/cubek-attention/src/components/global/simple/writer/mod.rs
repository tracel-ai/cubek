use cubecl::{self as cubecl, prelude::*, std::tensor::ViewMut};

use cubek_matmul::components::global::{GlobalWriterConfig, PartitionedStage, WriteEventListener};

mod plane;
mod unit;

use cubecl::std::tensor::layout::Coords2d;
pub use plane::*;
pub use unit::*;

use crate::components::stage::StageAttentionConfig;

#[cube]
pub trait AttentionWriter<'a, ES: Numeric, ESS: Size, EG: Numeric, EGS: Size>:
    WriteEventListener + 'a
{
    fn init<S: StageAttentionConfig>(
        global: ViewMut<'a, Vector<EG, EGS>, Coords2d>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    fn stage(&mut self) -> PartitionedStage<ES, ESS>;
}
