//! A conv launch's compile-time decisions.

use crate::kernels::conv::RingWriteBack;
use cubecl::client::Client;
use cubecl::tile::Launcher;
use cubek_microkernel::linear_recurrence::CausalConvPartitioning;
use cubek_microkernel::linear_recurrence::conv::cube::CubeQkNormedConvMicrokernel;
use cubek_std::routine::Grid;

/// What a conv launch compiles: the microkernel, the partitioning, the width the channels bind at,
/// and the launch that fills a prompt's rings after it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QkNormedConvBlueprint {
    pub microkernel: CubeQkNormedConvMicrokernel,
    pub partitioning: CausalConvPartitioning,
    pub width: usize,
    pub write_back: Option<RingWriteBack>,
}

impl QkNormedConvBlueprint {
    pub(crate) fn launcher(&self, client: &Client, grid: &Grid) -> Launcher {
        grid.launcher(client, || {
            (
                self.partitioning.kernel(),
                self.partitioning.as_cubek().space().clone(),
            )
        })
    }
}
