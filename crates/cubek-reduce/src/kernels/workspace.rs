//! The scratch a reduction's caller gives it ([`ReduceWorkspace`]): cubek allocates none.

use crate::kernels::{
    HypercubeLastArrivalReduceRoutine, ReduceElements, ReduceRoutineProblem, indexed,
};
use cubecl::client::Client;
use cubecl::server::Handle;
use cubecl::tile::launch::Decline;
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::Hardware;

/// Why the last-arrival hypercube declines a call whose caller gave it less scratch than it needs
/// ([`ReduceWorkspace::bytes_for`]).
pub const WORKSPACE_TOO_SMALL: Decline = Decline::Unserved {
    why: "the routine needs more scratch than the workspace its caller gave",
};

/// A launch whose caller's workspace is gone.
pub(crate) const WORKSPACE_LOST: Decline = Decline::Broken {
    condition: "prepare declines a launch whose caller gave no workspace",
    cause: None,
};

/// The scratch of a reduction whose last cube merges a row split across cubes, which its caller
/// allocates and owns: each cube's partial, its indices where the carrier keeps any, and one arrival counter a
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
    /// `with_indices`: enough for every setting of the last-arrival hypercube, at every width, so it
    /// serves whichever the strategy picks.
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};

    const SHAPE: [usize; 2] = [3, 70001];
    const AXIS: usize = 1;

    const BYTES: ReduceWorkspaceBytes = ReduceWorkspaceBytes {
        partials: 64,
        partial_indices: 32,
        arrivals: 8,
    };

    fn max_elements() -> ReduceElements {
        let f32 = ElemType::Float(FloatKind::F32);
        ReduceElements::for_operation(ReduceOperation::Max, f32, None)
    }

    /// What the last-arrival hypercube takes on the test device to reduce a [`SHAPE`] along
    /// [`AXIS`] by `operation`.
    fn bound(operation: ReduceOperation, with_indices: bool) -> ReduceWorkspaceBytes {
        let hw = Hardware::from_client(&cubecl::test_device().client());
        let problem =
            ReduceRoutineProblem::of_shape(&SHAPE, AXIS, operation, max_elements(), with_indices);
        HypercubeLastArrivalReduceRoutine::workspace_bound(&hw, &problem)
    }

    #[test]
    fn a_workspace_takes_what_the_last_arrival_hypercube_takes() {
        let bytes = ReduceWorkspace::bytes_for(
            &cubecl::test_device().client(),
            &SHAPE,
            AXIS,
            ReduceOperation::Max,
            max_elements(),
            false,
        );
        assert_eq!(bytes, bound(ReduceOperation::Max, false));
    }

    #[test]
    fn a_max_with_its_indices_takes_an_argmaxs_workspace() {
        let bytes = ReduceWorkspace::bytes_for(
            &cubecl::test_device().client(),
            &SHAPE,
            AXIS,
            ReduceOperation::Max,
            max_elements(),
            true,
        );
        assert_eq!(bytes, bound(ReduceOperation::ArgMax, true));
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn a_workspace_along_an_axis_past_the_rank_panics() {
        ReduceWorkspace::bytes_for(
            &cubecl::test_device().client(),
            &SHAPE,
            SHAPE.len(),
            ReduceOperation::Max,
            max_elements(),
            false,
        );
    }

    #[test]
    fn a_workspace_is_as_large_as_its_buffers() {
        let client = cubecl::test_device().client();
        let workspace = ReduceWorkspace {
            partials: client.empty(BYTES.partials as usize),
            partial_indices: client.empty(BYTES.partial_indices as usize),
            arrivals: client.empty(BYTES.arrivals as usize),
        };
        assert_eq!(workspace.bytes(), BYTES);
    }

    #[test]
    fn bytes_hold_as_many() {
        assert!(BYTES.holds(&BYTES));
    }

    #[test]
    fn fewer_partials_than_needed_hold_nothing() {
        let given = ReduceWorkspaceBytes {
            partials: BYTES.partials - 1,
            ..ReduceWorkspaceBytes::UNBOUNDED
        };
        assert!(!given.holds(&BYTES));
    }

    #[test]
    fn fewer_partial_indices_than_needed_hold_nothing() {
        let given = ReduceWorkspaceBytes {
            partial_indices: BYTES.partial_indices - 1,
            ..ReduceWorkspaceBytes::UNBOUNDED
        };
        assert!(!given.holds(&BYTES));
    }

    #[test]
    fn fewer_arrivals_than_needed_hold_nothing() {
        let given = ReduceWorkspaceBytes {
            arrivals: BYTES.arrivals - 1,
            ..ReduceWorkspaceBytes::UNBOUNDED
        };
        assert!(!given.holds(&BYTES));
    }

    #[test]
    fn the_max_takes_the_larger_of_each_buffer() {
        let other = ReduceWorkspaceBytes {
            partials: BYTES.partials * 2,
            partial_indices: BYTES.partial_indices / 2,
            arrivals: BYTES.arrivals * 2,
        };
        assert_eq!(
            BYTES.max(other),
            ReduceWorkspaceBytes {
                partials: other.partials,
                partial_indices: BYTES.partial_indices,
                arrivals: other.arrivals,
            }
        );
    }
}
