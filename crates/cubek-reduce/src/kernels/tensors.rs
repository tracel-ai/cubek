//! How a reduce launch reads its operands and binds them.

use crate::kernels::{
    REDUCED, ReduceBlueprint, ReduceElements, ReduceRoutineProblem, ReduceWorkspace, SLOTS,
};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::launch::{Bound, Decline, Unbound};
use cubecl::tile::{Dimension, Geometry, Launcher};
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

#[cfg(test)]
use crate::kernels::testing::zeros;

/// Why an unbound reduction declines where its microkernel reads narrower lines than the caller's
/// input is stated at ([`ReduceTensors::unbound`]): the caller computes its lines whole.
pub const UNBOUND_LINES_NARROWER: Decline = Decline::Unserved {
    why: "the microkernel reads narrower lines than the caller's input is stated at",
};

/// The input and the output of one reduction, and the problem they state: the input a buffer
/// ([`TensorBinding`]) or the geometry of an input the caller binds once the launch is prepared
/// ([`Geometry`], [`ReduceCandidate::launch_unbound`](crate::kernels::ReduceCandidate::launch_unbound)),
/// the output a buffer.
///
/// Both are read where they lie, at their own strides: each binds its axes onto the problem's
/// dimensions, so a reduction along any axis of a tensor of any rank binds without a copy.
#[derive(Clone)]
pub struct ReduceTensors<In = TensorBinding> {
    client: Client,
    input: In,
    out: TensorBinding,
    /// The indices written beside the values, of `out`'s shape at their own strides, where they
    /// are.
    indices: Option<TensorBinding>,
    /// The scratch the caller gave, where it gave any.
    workspace: Option<ReduceWorkspace>,
    problem: ReduceRoutineProblem,
}

impl<In> core::fmt::Debug for ReduceTensors<In> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReduceTensors")
            .field("problem", &self.problem)
            .finish_non_exhaustive()
    }
}

impl<In> ReduceTensors<In> {
    pub fn problem(&self) -> &ReduceRoutineProblem {
        &self.problem
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// These tensors with `workspace`, the last-arrival hypercube's scratch, which holds at least
    /// the bytes [`ReduceWorkspace::bytes_for`] states for this reduction: without one, it declines,
    /// and a smaller one is outside the reduction's Domain.
    pub fn with_workspace(self, workspace: ReduceWorkspace) -> Self {
        Self {
            problem: ReduceRoutineProblem {
                partials: Some(workspace.partials_in(self.problem.elements.accumulator)),
                ..self.problem
            },
            workspace: Some(workspace),
            ..self
        }
    }
}

impl ReduceTensors<TensorBinding> {
    /// `out = operation(input)` along axis `axis`: `out` has `input`'s shape, `axis` one wide, or
    /// `k` wide for a top-k.
    ///
    /// # Panics
    ///
    /// Where `axis` is not an axis of `input`, as of one that has none, or `out`'s shape is not the
    /// one above.
    pub fn new(
        client: &Client,
        input: TensorBinding,
        out: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Self {
        let problem = ReduceRoutineProblem::of_buffers(&input, &out, axis, operation, elements);
        Self {
            client: client.clone(),
            input,
            out,
            indices: None,
            workspace: None,
            problem,
        }
    }

    /// `values = operation(input)` along axis `axis`, and the indices of those values to
    /// `indices`, in `index_element`, from one launch: `indices` has `values`' shape, each written
    /// at its own strides.
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and where the operation's carrier keeps no index (a sum's, or a
    /// max's, whose pair an argmax writes), or `indices` does not have `values`' shape.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and what the reduction computes, as a caller states them"
    )]
    pub fn with_indices(
        client: &Client,
        input: TensorBinding,
        values: TensorBinding,
        indices: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
        index_element: cubecl::ir::ElemType,
    ) -> Self {
        assert!(
            operation.carrier().carries_indices(),
            "{operation:?} accumulates values without their indices"
        );
        assert!(
            indices.shape == values.shape,
            "the indices have the values' shape"
        );
        let mut tensors = Self::new(client, input, values, axis, operation, elements);
        tensors.problem = tensors.problem.writing_indices(index_element, &indices);
        tensors.indices = Some(indices);
        tensors
    }
}

impl ReduceTensors<Geometry> {
    /// `out = operation(input)` along dim `axis`, the input laid out as `input` but bound by the
    /// caller once the launch is prepared and read at exactly `lane_count`: what a fused read stands
    /// for, whose lines the caller's code computes whole.
    ///
    /// # Panics
    ///
    /// As [`new`](ReduceTensors::new).
    pub fn unbound(
        client: &Client,
        input: Geometry,
        lane_count: usize,
        out: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Self {
        let problem =
            ReduceRoutineProblem::of_unbound(&input, lane_count, &out, axis, operation, elements);
        Self {
            client: client.clone(),
            input,
            out,
            indices: None,
            workspace: None,
            problem,
        }
    }
}

impl ReduceTensors<TensorBinding> {
    /// Both operands bound as `blueprint` states them, on `grid`.
    ///
    /// # Panics
    ///
    /// Where `blueprint` and `grid` were not prepared for these tensors' problem.
    pub(crate) fn bind<M>(&self, blueprint: &ReduceBlueprint<M>, grid: &Grid) -> ReduceBinding {
        ReduceBinding::bound(self, blueprint, grid)
    }
}

impl ReduceTensors<Geometry> {
    /// The problem as the input's own geometry states it, its alignment the geometry's rather than
    /// the lanes the caller binds it at.
    pub(crate) fn stated_problem(&self) -> ReduceRoutineProblem {
        self.problem.with_input_aligned_as(&self.input)
    }

    /// The input's spec and geometry as `blueprint` states them, on `grid`, for the caller to bind,
    /// and the output bound. It panics where binding both operands does.
    pub(crate) fn bind<M>(
        &self,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> ReduceBinding<Unbound, Bound> {
        ReduceBinding::unbound(self, blueprint, grid)
    }
}

/// The operands of one launch and the launcher they were bound through: each a [`Bound`] buffer,
/// an [`Unbound`] spec the caller binds, or the launch argument either became.
pub(crate) struct ReduceBinding<In = Bound, Out = In> {
    pub launcher: Launcher,
    pub input: In,
    pub out: Out,
    /// The indices beside the values, where the launch writes both.
    pub indices: Option<Bound>,
    pub workspace: Option<ReduceWorkspace>,
}

impl<In, Out> ReduceBinding<In, Out> {
    /// The input and the output made into what `input` and `out` make of them.
    pub fn map<I, O>(
        self,
        input: impl FnOnce(In) -> I,
        out: impl FnOnce(Out) -> O,
    ) -> ReduceBinding<I, O> {
        ReduceBinding {
            launcher: self.launcher,
            input: input(self.input),
            out: out(self.out),
            indices: self.indices,
            workspace: self.workspace,
        }
    }
}

impl ReduceBinding {
    /// The operands of `tensors`, and their indices where they have any, bound as `blueprint` states
    /// them, on `grid`.
    fn bound<M>(tensors: &ReduceTensors, blueprint: &ReduceBlueprint<M>, grid: &Grid) -> Self {
        let problem = &tensors.problem;
        let launcher = launcher_of(&tensors.client, problem, blueprint, grid);
        let (input, dimensions) = problem.labelled(tensors.input.clone(), REDUCED);
        let input = launcher
            .arg(input, problem.elements.input)
            .dimensions(&dimensions)
            .boundary(problem.input_boundary())
            .vectorize(blueprint.lane_count)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let indices = tensors
            .indices
            .as_ref()
            .zip(problem.indices)
            .map(|(indices, stated)| {
                out_bound(&launcher, indices, stated.element, problem, blueprint)
            });
        ReduceBinding {
            input,
            out: out_bound(
                &launcher,
                &tensors.out,
                problem.elements.out,
                problem,
                blueprint,
            ),
            indices,
            workspace: tensors.workspace.clone(),
            launcher,
        }
    }
}

impl ReduceBinding<Unbound, Bound> {
    /// The spec and geometry of the input of `tensors`, which the caller binds, as `blueprint`
    /// states them, on `grid`, and their output bound.
    fn unbound<M>(
        tensors: &ReduceTensors<Geometry>,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Self {
        let problem = &tensors.problem;
        let launcher = launcher_of(&tensors.client, problem, blueprint, grid);
        let (input, dimensions) = problem.labelled_geometry(&tensors.input, REDUCED);
        let input = launcher
            .unbound(&input)
            .dimensions(&dimensions)
            .boundary(problem.input_boundary())
            .vectorize(blueprint.lane_count)
            .build_spec()
            .expect(ADMITTED_PLAN_RUNS);
        ReduceBinding {
            input,
            out: out_bound(
                &launcher,
                &tensors.out,
                problem.elements.out,
                problem,
                blueprint,
            ),
            indices: None,
            workspace: tensors.workspace.clone(),
            launcher,
        }
    }
}

/// The launcher `grid` builds for `problem` placed by `blueprint`, over the extents its partitioning
/// states, which the kernel leaves to the launch.
fn launcher_of<M>(
    client: &Client,
    problem: &ReduceRoutineProblem,
    blueprint: &ReduceBlueprint<M>,
    grid: &Grid,
) -> Launcher {
    grid.launcher(client, || {
        (
            problem.kernel(&blueprint.partitioning),
            blueprint.partitioning.space().clone(),
        )
    })
}

/// `buffer`, an output of `problem` or the indices beside it, bound in `element` at the output's
/// lanes.
fn out_bound<M>(
    launcher: &Launcher,
    buffer: &TensorBinding,
    element: ElemType,
    problem: &ReduceRoutineProblem,
    blueprint: &ReduceBlueprint<M>,
) -> Bound {
    bound_at(
        launcher,
        problem,
        buffer.clone(),
        SLOTS,
        element,
        blueprint.out_lane_count,
    )
}

/// `buffer`, one a launch of `problem` writes holding `along` in place of the reduced axis, bound
/// in `element` at `lane_count` ([`ReduceRoutineProblem::labelled`]).
///
/// # Panics
///
/// Where the launch the launcher was prepared for cannot bind it, which its routine's prepare
/// rules out.
pub(crate) fn bound_at(
    launcher: &Launcher,
    problem: &ReduceRoutineProblem,
    buffer: TensorBinding,
    along: Dimension,
    element: ElemType,
    lane_count: usize,
) -> Bound {
    let (buffer, dimensions) = problem.labelled(buffer, along);
    launcher
        .arg(buffer, element)
        .dimensions(&dimensions)
        .vectorize(lane_count)
        .build()
        .expect(ADMITTED_PLAN_RUNS)
}

#[cfg(test)]
impl ReduceTensors {
    /// A tensor of `shape` in f32 reduced along `axis` by `operation` on the test device, holding
    /// zeros as its output does.
    pub(crate) fn zeros(shape: &[usize], axis: usize, operation: ReduceOperation) -> Self {
        let client = cubecl::test_device().client();
        let problem = ReduceRoutineProblem::dense(shape, axis, operation);
        let out_shape = problem.shape_with_reduced_count(operation.written_per_row());
        Self::new(
            &client,
            zeros(&client, shape),
            zeros(&client, &out_shape),
            axis,
            operation,
            problem.elements,
        )
    }
}

#[cfg(test)]
impl ReduceTensors<Geometry> {
    /// [`zeros`](ReduceTensors::zeros), its input a dense one the caller binds a value a line.
    pub(crate) fn unbound_zeros(shape: &[usize], axis: usize, operation: ReduceOperation) -> Self {
        let bound = ReduceTensors::zeros(shape, axis, operation);
        Self::unbound(
            &bound.client,
            Geometry::from(&bound.input),
            1,
            bound.out,
            axis,
            operation,
            bound.problem.elements,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{FloatKind, IntKind};

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const INPUT_SHAPE: [usize; 2] = [4, 8];
    const VALUES_SHAPE: [usize; 2] = [4, 1];

    /// `ReduceTensors::with_indices` for `operation` along the last axis of an [`INPUT_SHAPE`]
    /// input, the values of [`VALUES_SHAPE`] and the indices of `indices_shape`.
    fn tensors_with_indices(operation: ReduceOperation, indices_shape: &[usize]) -> ReduceTensors {
        let client = cubecl::test_device().client();
        ReduceTensors::with_indices(
            &client,
            zeros(&client, &INPUT_SHAPE),
            zeros(&client, &VALUES_SHAPE),
            zeros(&client, indices_shape),
            VALUES_SHAPE.len() - 1,
            operation,
            ReduceElements {
                input: F32,
                accumulator: F32,
                out: F32,
            },
            ElemType::Int(IntKind::I32),
        )
    }

    #[test]
    #[should_panic(expected = "accumulates values without their indices")]
    fn indices_beside_a_sum_panic() {
        tensors_with_indices(ReduceOperation::Sum, &VALUES_SHAPE);
    }

    #[test]
    #[should_panic(expected = "the indices have the values' shape")]
    fn indices_of_another_shape_panic() {
        tensors_with_indices(ReduceOperation::ArgMax, &[1, 4]);
    }
}
