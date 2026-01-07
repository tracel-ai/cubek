use crate::{
    components::{
        global::{
            PlaneFlowPartitionRule, WriteEventListener, WriteTiling, memory::GlobalMemoryConfig,
        },
        stage::{Stage, StageFamily, StageMemoryConfig},
    },
    definition::MatrixPrecision,
};
use cubecl::prelude::*;
use cubecl::std::tensor::{View, layout::Coords2d};

pub trait GlobalWriterFamily: 'static + Send + Sync {
    type Stage: StageFamily<ReadWrite>;
    type Writer<IP: MatrixPrecision>: GlobalWriter<
            IP,
            Stage = <Self::Stage as StageFamily<ReadWrite>>::Stage<IP::Stage, WriteTiling>,
        >;
}

#[cube]
/// Responsible of writing the accumulated stage matmul output
/// to global memory
pub trait GlobalWriter<IP: MatrixPrecision>:
    WriteEventListener + CubeType + 'static + Send + Sync
{
    /// Tile stage that stores the data for this writer
    type Stage: Stage<IP::Stage, ReadWrite>;

    /// Init this writer from a global tensor and config
    fn init(
        tensor: View<Line<IP::Global>, Coords2d, ReadWrite>,
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
