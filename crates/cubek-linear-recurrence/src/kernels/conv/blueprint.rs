//! A conv launch's compile-time decisions.

use crate::kernels::conv::RingWriteBack;
use cubecl::client::Client;
use cubecl::tile::Launcher;
use cubek_microkernel::linear_recurrence::CausalConvPartitioning;
use cubek_std::routine::Grid;

/// What a launch of a conv compiles: the microkernel, the partitioning that places its cubes, the
/// width the channels bind at, and the launch that fills a prompt's rings after it.
///
/// One type for every routine of this kind: what differs between them is the microkernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalConvBlueprint<M> {
    pub microkernel: M,
    pub partitioning: CausalConvPartitioning,
    pub width: usize,
    pub write_back: Option<RingWriteBack>,
}

impl<M> CausalConvBlueprint<M> {
    /// `microkernel` placed by `partitioning`, the channels bound at `width`, a prompt's rings
    /// filled by `write_back`.
    pub fn new(
        microkernel: M,
        partitioning: CausalConvPartitioning,
        width: usize,
        write_back: Option<RingWriteBack>,
    ) -> Self {
        Self {
            microkernel,
            partitioning,
            width,
            write_back,
        }
    }

    /// The launcher of `grid`, settled over the partitioning's extents on its first launch.
    pub(crate) fn launcher(&self, client: &Client, grid: &Grid) -> Launcher {
        grid.launcher(client, || {
            (
                self.partitioning.kernel(),
                self.partitioning.as_cubek().space().clone(),
            )
        })
    }
}
