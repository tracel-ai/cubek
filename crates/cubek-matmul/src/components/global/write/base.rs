use crate::{
    components::{
        global::{
            PlaneFlowPartitionRule, WriteEventListener, WriteTiling, memory::GlobalMemoryConfig,
        },
        stage::{Stage, StageFamily},
    },
    definition::{self, MatrixTypes},
};
use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::Coords2d},
};
use cubek_std::stage::StageMemoryConfig;

pub type WriterStage<GW, MT> = <<GW as GlobalWriterFamily>::Stage as StageFamily>::Stage<
    definition::Stage<MT>,
    definition::StageSize<MT>,
    WriteTiling,
>;

pub trait GlobalWriterFamily: 'static + Send + Sync {
    type Stage: StageFamily;
    type Writer<'a, IP: MatrixTypes>: GlobalWriter<
            'a,
            IP,
            Stage = <Self::Stage as StageFamily>::Stage<IP::Stage, IP::StageSize, WriteTiling>,
        >;
}

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<'a, IP: MatrixTypes>: WriteEventListener + CubeType + 'a {
    /// Tile stage that stores the data for this writer
    type Stage: Stage<IP::Stage>;

    /// Init this writer from a global tensor and config
    fn init(
        tensor: ViewMut<'a, Vector<IP::Global, IP::GlobalSize>, Coords2d>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self;

    /// Stage used by this writer
    fn stage(this: &Self) -> Self::Stage;
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalWriterConfig {
    pub gmem_config: GlobalMemoryConfig,
    pub smem_config: StageMemoryConfig,
    pub plane_flow_partition_rule: PlaneFlowPartitionRule,
    pub plane_dim: u32,
}
