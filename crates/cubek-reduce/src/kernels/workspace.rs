//! The scratch a reduction's caller gives it ([`ReduceWorkspace`]): cubek allocates none.

use crate::kernels::{
    HypercubeLastArrivalReduceRoutine, ReduceElements, ReducePartials, ReduceRoutineProblem,
    indexed,
};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::server::Handle;
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::Hardware;

/// What should hold of a launch that finds its caller's workspace gone.
pub(crate) const WORKSPACE_CHECKED_BY_PREPARE: &str =
    "prepare declines a launch whose caller gave no workspace";

/// The scratch of a reduction whose last cube merges a row split across cubes, which its caller
/// allocates and owns: each cube's partial, its indices where the carrier keeps any, and one arrival
/// counter a box of rows.
///
/// # Assumptions
///
/// - Its buffers live on the client the tensors it is given with launch on.
/// - Every counter of `arrivals` holds zero when the caller first gives the workspace.
/// - The launches given one workspace run one after another, each once the one before it ends.
/// - Its three buffers alias neither each other nor the input or the outputs of a launch given
///   them.
///
/// # Guarantees
///
/// - Every launch leaves the counters at zero, so one workspace serves every launch after it.
#[derive(Debug, Clone)]
pub struct ReduceWorkspace {
    /// The partials, in the accumulator's element.
    pub partials: Handle,
    /// The partials' indices, in `u32`.
    pub partial_indices: Handle,
    /// The arrival counters, `u32` each, zero.
    pub arrivals: Handle,
}

/// The bytes each buffer of a [`ReduceWorkspace`] must hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReduceWorkspaceBytes {
    pub partials_bytes: u64,
    pub partial_indices_bytes: u64,
    pub arrivals_bytes: u64,
}

impl ReduceWorkspace {
    /// The bytes a workspace takes for reducing an input of `shape` along `axis` by `operation`
    /// in `elements` on `client`'s device, with the indices written beside the values where
    /// `with_indices`: enough for every setting of the last-arrival hypercube, at every width, so it
    /// serves whichever the strategy picks.
    ///
    /// # Assumptions
    ///
    /// - `axis` is an axis of `shape`.
    ///
    /// # Guarantees
    ///
    /// - The bytes serve an input of `shape` in any layout: the partials are laid out by the
    ///   shape alone, whatever the input's strides.
    ///
    /// # Panics
    ///
    /// Where its Assumption is broken.
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
        let partials = HypercubeLastArrivalReduceRoutine::partials_bound(
            &Hardware::from_client(client),
            &problem,
        );
        ReduceWorkspaceBytes {
            partials_bytes: partials.value_count as u64 * elements.accumulator.size() as u64,
            partial_indices_bytes: partials.index_count as u64 * size_of::<u32>() as u64,
            arrivals_bytes: partials.arrival_count as u64 * size_of::<u32>() as u64,
        }
    }
}

impl ReduceWorkspace {
    /// The cells its buffers hold, its partials in `accumulator`.
    pub(crate) fn partials_in(&self, accumulator: ElemType) -> ReducePartials {
        ReducePartials {
            value_count: (self.partials.size_in_used() / accumulator.size() as u64) as usize,
            index_count: (self.partial_indices.size_in_used() / size_of::<u32>() as u64) as usize,
            arrival_count: (self.arrivals.size_in_used() / size_of::<u32>() as u64) as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::FloatKind;

    const SHAPE: [usize; 2] = [3, 70001];
    const AXIS: usize = 1;

    fn max_elements() -> ReduceElements {
        let f32 = ElemType::Float(FloatKind::F32);
        ReduceElements::for_operation(ReduceOperation::Max, f32, None)
    }

    /// The bytes of the cells the last-arrival hypercube writes on the test device, four a cell, to
    /// reduce a [`SHAPE`] along [`AXIS`] by `operation`.
    fn bound(operation: ReduceOperation, with_indices: bool) -> ReduceWorkspaceBytes {
        let hw = Hardware::from_client(&cubecl::test_device().client());
        let problem =
            ReduceRoutineProblem::of_shape(&SHAPE, AXIS, operation, max_elements(), with_indices);
        let partials = HypercubeLastArrivalReduceRoutine::partials_bound(&hw, &problem);
        ReduceWorkspaceBytes {
            partials_bytes: 4 * partials.value_count as u64,
            partial_indices_bytes: 4 * partials.index_count as u64,
            arrivals_bytes: 4 * partials.arrival_count as u64,
        }
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
    fn a_workspace_holds_as_many_cells_as_its_buffers() {
        let client = cubecl::test_device().client();
        let workspace = ReduceWorkspace {
            partials: client.empty(64),
            partial_indices: client.empty(32),
            arrivals: client.empty(8),
        };
        let partials = workspace.partials_in(ElemType::Float(FloatKind::F64));
        assert_eq!(
            partials,
            ReducePartials {
                value_count: 8,
                index_count: 8,
                arrival_count: 2,
            }
        );
    }
}
