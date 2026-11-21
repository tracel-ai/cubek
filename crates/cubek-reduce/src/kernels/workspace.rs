//! The scratch a reduction's caller gives it ([`ReduceWorkspace`]): cubek allocates none.

use crate::kernels::{
    HypercubeLastArrivalReduceRoutine, ReduceElements, ReduceRoutineProblem, indexed,
};
use cubecl::client::Client;
use cubecl::server::Handle;
use cubecl::tile::launch::Decline;
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::Hardware;

/// Why a routine that splits a row across cubes declines a call whose caller gave it less scratch
/// than it needs ([`ReduceWorkspace::bytes_for`]).
pub const WORKSPACE_TOO_SMALL: Decline = Decline::Unserved {
    why: "the routine needs more scratch than the workspace its caller gave",
};

/// The scratch of a reduction that splits its rows across cubes, which its caller allocates and
/// owns: each cube's partial, its indices where the carrier keeps any, and one arrival counter a
/// box of rows. The counters hold zero when the caller first gives them, and every launch leaves
/// them at zero, so one workspace serves every launch that runs after the one before it ends.
#[derive(Debug, Clone)]
pub struct ReduceWorkspace {
    /// The partials, in the accumulator's element.
    pub partials: Handle,
    /// The partials' indices, in `u32`.
    pub partial_indices: Handle,
    /// The arrival counters, `u32` each, zero.
    pub arrivals: Handle,
}

/// The bytes each buffer of a [`ReduceWorkspace`] holds, or must hold.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReduceWorkspaceBytes {
    pub partials: u64,
    pub partial_indices: u64,
    pub arrivals: u64,
}

impl ReduceWorkspace {
    /// The bytes a workspace takes for reducing an input of `shape` along `axis` by `operation`
    /// in `elements` on `client`'s device, with the indices written beside the values where
    /// `with_indices`: enough for every setting of every routine that splits a row across cubes,
    /// at every width, so it serves whichever the strategy picks.
    ///
    /// # Panics
    ///
    /// Where `axis` is not an axis of `shape`.
    pub fn bytes_for(
        client: &Client,
        shape: &[usize],
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
        with_indices: bool,
    ) -> ReduceWorkspaceBytes {
        // As `execute_with_indices` writes them: a max with its indices is an arg max.
        let operation = match with_indices {
            true => indexed(operation).unwrap_or(operation),
            false => operation,
        };
        let problem =
            ReduceRoutineProblem::of_shape(shape, axis, operation, elements, with_indices);
        HypercubeLastArrivalReduceRoutine::workspace_bound(&Hardware::from_client(client), &problem)
    }

    /// The bytes each of its buffers holds.
    pub fn bytes(&self) -> ReduceWorkspaceBytes {
        ReduceWorkspaceBytes {
            partials: self.partials.size_in_used(),
            partial_indices: self.partial_indices.size_in_used(),
            arrivals: self.arrivals.size_in_used(),
        }
    }
}

impl ReduceWorkspaceBytes {
    /// Whether each buffer holds at least what `needs` asks of it.
    pub fn holds(&self, needs: &ReduceWorkspaceBytes) -> bool {
        self.partials >= needs.partials
            && self.partial_indices >= needs.partial_indices
            && self.arrivals >= needs.arrivals
    }

    /// What a key's representative is given: as much as any of its calls asks.
    pub const UNBOUNDED: Self = Self {
        partials: u64::MAX,
        partial_indices: u64::MAX,
        arrivals: u64::MAX,
    };

    /// Each buffer the larger of this and `other`.
    pub fn max(self, other: Self) -> Self {
        Self {
            partials: self.partials.max(other.partials),
            partial_indices: self.partial_indices.max(other.partial_indices),
            arrivals: self.arrivals.max(other.arrivals),
        }
    }
}
