//! A causal conv launch's compile-time decisions.

use crate::kernels::conv::RingWriteBack;
use cubecl::client::Client;
use cubecl::tile::Launcher;
use cubek_microkernel::linear_recurrence::CausalConvPartitioning;
use cubek_microkernel::linear_recurrence::conv::cube::CubeCausalConvMicrokernel;
use cubek_std::routine::Grid;

/// What a causal conv launch compiles: the microkernel, the partitioning, and the width the
/// channels bind at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalConvBlueprint {
    pub microkernel: CubeCausalConvMicrokernel,
    pub partitioning: CausalConvPartitioning,
    pub width: usize,
    pub write_back: Option<RingWriteBack>,
}

impl CausalConvBlueprint {
    pub(crate) fn launcher(&self, client: &Client, grid: &Grid) -> Launcher {
        grid.launcher(client, || {
            (
                self.partitioning.kernel(),
                self.partitioning.as_cubek().space().clone(),
            )
        })
    }
}
