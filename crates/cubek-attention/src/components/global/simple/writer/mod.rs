use cubecl::prelude::*;
use cubecl::{self as cubecl};

use cubek_matmul::components::global::{GlobalWriterConfig, PartitionedStage, WriteEventListener};

mod plane;
mod unit;

use cubecl::std::tensor::{View, layout::Coords2d};
pub use plane::*;
pub use unit::*;

use crate::components::stage::StageAttentionConfig;

#[cube]
pub trait AttentionWriter<ES: Numeric, EG: Numeric>: WriteEventListener {
    fn init<S: StageAttentionConfig>(
        global: View<Line<EG>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    fn stage(&mut self) -> PartitionedStage<ES>;
}
