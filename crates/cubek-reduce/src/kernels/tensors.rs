//! How a reduce launch reads its operands and binds them.

use crate::ReduceError;
use crate::kernels::{
    REDUCED, ReduceBlueprint, ReduceElements, ReduceRoutineProblem, ReduceWorkspace, SLOTS,
    indexed, validate_axis, validate_indices, validate_shape,
};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::launch::{Bound, Decline, Unbound};
use cubecl::tile::{Dimension, Geometry, Launcher};
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Breakable, Grid};

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
        fmt_problem(&self.problem, f)
    }
}

/// `problem` written as the tensors that state it, whose buffers it names none of.
fn fmt_problem(
    problem: &ReduceRoutineProblem,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    f.debug_struct("ReduceTensors")
        .field("problem", problem)
        .finish_non_exhaustive()
}

impl<In> ReduceTensors<In> {
    pub fn problem(&self) -> &ReduceRoutineProblem {
        &self.problem
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

impl<In> ReduceTensors<In> {
    /// The launcher `blueprint` launches on `grid` through, and the output bound as it states it.
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] where the tile engine refuses what the routine prepared.
    pub(crate) fn bind_out<M>(
        &self,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Result<(Launcher, Bound), Decline> {
        bound_out(self, blueprint, grid)
    }
}

impl<In: Clone> ReduceTensors<In> {
    /// These tensors with `workspace`, the last-arrival hypercube's scratch, which declines
    /// them without one.
    pub fn with_workspace(&self, workspace: ReduceWorkspace) -> Self {
        Self {
            problem: ReduceRoutineProblem {
                partials: Some(workspace.partials_in(self.problem.elements.accumulator)),
                ..self.problem.clone()
            },
            workspace: Some(workspace),
            ..self.clone()
        }
    }
}

impl ReduceTensors<TensorBinding> {
    /// `out = operation(input)` along axis `axis`.
    ///
    /// # Assumptions
    ///
    /// - `axis` is an axis of `input`, which has one at least.
    /// - `out` has `input`'s shape, `axis` one wide, or `k` wide for a top-k.
    ///
    /// # Panics
    ///
    /// Where an Assumption is broken.
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
    /// `indices`, in `index_element`, from one launch, each written at its own strides.
    ///
    /// # Assumptions
    ///
    /// - Those of [`new`](Self::new), `values` its `out`.
    /// - The operation's carrier keeps indices: not a sum's, nor a max's, whose pair an argmax
    ///   writes.
    /// - `indices` has `values`' shape.
    /// - `index_element` counts every position of the reduced axis.
    ///
    /// # Panics
    ///
    /// Where an Assumption is broken.
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

    /// [`new`](Self::new), the call first checked to lie in the reduction's Domain: what
    /// [`Reduce::execute`](crate::kernels::Reduce::execute) launches on, for a caller that picks the
    /// kernel itself.
    ///
    /// # Errors
    ///
    /// As [`Reduce::execute`](crate::kernels::Reduce::execute), but for a kernel's decline.
    pub fn checked(
        client: &Client,
        input: TensorBinding,
        out: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Result<Self, ReduceError> {
        validate_axis(input.shape.len(), axis)?;
        validate_shape(&input.shape, &out.shape, axis, operation.written_per_row())?;
        Ok(Self::new(client, input, out, axis, operation, elements))
    }

    /// [`with_indices`](Self::with_indices), the call first checked to lie in the reduction's
    /// Domain, `Max`, `Min` and `TopK` writing the same pair as `ArgMax`, `ArgMin` and `ArgTopK`:
    /// what [`Reduce::execute_with_indices`](crate::kernels::Reduce::execute_with_indices)
    /// launches on.
    ///
    /// # Errors
    ///
    /// As [`Reduce::execute_with_indices`](crate::kernels::Reduce::execute_with_indices), but for a
    /// kernel's decline.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands and what the reduction computes, as a caller states them"
    )]
    pub fn checked_with_indices(
        client: &Client,
        input: TensorBinding,
        values: TensorBinding,
        indices: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
        index_element: ElemType,
    ) -> Result<Self, ReduceError> {
        let operation = indexed(operation)?;
        validate_axis(input.shape.len(), axis)?;
        validate_shape(
            &input.shape,
            &values.shape,
            axis,
            operation.written_per_row(),
        )?;
        validate_indices(&values.shape, &indices.shape)?;
        Ok(Self::with_indices(
            client,
            input,
            values,
            indices,
            axis,
            operation,
            elements,
            index_element,
        ))
    }
}

impl ReduceTensors<Geometry> {
    /// `out = operation(input)` along dim `axis`, the input laid out as `input` but bound by the
    /// caller once the launch is prepared and read at exactly `lane_count`: what a fused read stands
    /// for, whose lines the caller's code computes whole.
    ///
    /// # Assumptions
    ///
    /// - Those of [`new`](ReduceTensors::new), of `input`'s geometry.
    /// - The caller binds an input laid out as `input`, read at exactly `lane_count`.
    ///
    /// # Panics
    ///
    /// Where an Assumption is broken.
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
    /// Both operands bound as `blueprint` states them, on `grid`, and the launcher they were bound
    /// through.
    ///
    /// # Errors
    ///
    /// As [`bind_out`](Self::bind_out).
    pub(crate) fn bind<M>(
        &self,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Result<(Launcher, ReduceBinding), Decline> {
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
    /// the output bound, and the launcher it was bound through.
    ///
    /// # Errors
    ///
    /// Where binding both operands errs.
    pub(crate) fn bind<M>(
        &self,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Result<(Launcher, ReduceBinding<Unbound, Bound>), Decline> {
        ReduceBinding::unbound(self, blueprint, grid)
    }
}

/// The operands of one launch: each a [`Bound`] buffer, an [`Unbound`] spec the caller binds, or
/// the launch argument either became.
pub(crate) struct ReduceBinding<In = Bound, Out = In> {
    pub input: In,
    pub out: Out,
    /// The indices beside the values, where the launch writes both.
    pub indices: Option<Bound>,
    pub workspace: Option<ReduceWorkspace>,
}

/// `binding`, its input and output made into what `input` and `out` make of them.
pub(crate) fn bound_as<In, Out, I, O>(
    binding: ReduceBinding<In, Out>,
    input: impl FnOnce(In) -> I,
    out: impl FnOnce(Out) -> O,
) -> ReduceBinding<I, O> {
    ReduceBinding {
        input: input(binding.input),
        out: out(binding.out),
        indices: binding.indices,
        workspace: binding.workspace,
    }
}

/// The launcher and the output [`ReduceTensors::bind_out`] answers.
fn bound_out<In, M>(
    tensors: &ReduceTensors<In>,
    blueprint: &ReduceBlueprint<M>,
    grid: &Grid,
) -> Result<(Launcher, Bound), Decline> {
    let problem = &tensors.problem;
    let launcher = launcher_of(&tensors.client, problem, blueprint, grid)?;
    let out = output_bound(
        &launcher,
        &tensors.out,
        problem.elements.out,
        problem,
        blueprint,
    )?;
    Ok((launcher, out))
}

impl ReduceBinding {
    /// The operands of `tensors`, and their indices where they have any, bound as `blueprint` states
    /// them, on `grid`, and the launcher they were bound through.
    fn bound<M>(
        tensors: &ReduceTensors,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Result<(Launcher, Self), Decline> {
        let problem = &tensors.problem;
        let (launcher, out) = bound_out(tensors, blueprint, grid)?;
        let (input, dimensions) = problem.labelled(tensors.input.clone(), REDUCED);
        let input = launcher
            .arg(input, problem.elements.input)
            .dimensions(&dimensions)
            .boundary(problem.input_boundary())
            .vectorize(blueprint.lane_count)
            .build()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        let indices = tensors
            .indices
            .as_ref()
            .zip(problem.indices)
            .map(|(indices, stated)| {
                output_bound(&launcher, indices, stated.element, problem, blueprint)
            })
            .transpose()?;
        let binding = ReduceBinding {
            input,
            out,
            indices,
            workspace: tensors.workspace.clone(),
        };
        Ok((launcher, binding))
    }
}

impl ReduceBinding<Unbound, Bound> {
    /// The spec and geometry of the input of `tensors`, which the caller binds, as `blueprint`
    /// states them, on `grid`, their output bound, and the launcher it was bound through.
    fn unbound<M>(
        tensors: &ReduceTensors<Geometry>,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Result<(Launcher, Self), Decline> {
        let problem = &tensors.problem;
        let (launcher, out) = bound_out(tensors, blueprint, grid)?;
        let (input, dimensions) = problem.labelled_geometry(&tensors.input, REDUCED);
        let input = launcher
            .unbound(&input)
            .dimensions(&dimensions)
            .boundary(problem.input_boundary())
            .vectorize(blueprint.lane_count)
            .build_spec()
            .or_broken(ADMITTED_PLAN_RUNS)?;
        let binding = ReduceBinding {
            input,
            out,
            indices: None,
            workspace: tensors.workspace.clone(),
        };
        Ok((launcher, binding))
    }
}

/// The launcher `grid` builds for `problem` placed by `blueprint`, over the extents its partitioning
/// states, which the kernel leaves to the launch.
fn launcher_of<M>(
    client: &Client,
    problem: &ReduceRoutineProblem,
    blueprint: &ReduceBlueprint<M>,
    grid: &Grid,
) -> Result<Launcher, Decline> {
    grid.launcher(client, || {
        (
            problem.kernel(&blueprint.partitioning),
            blueprint.partitioning.space().clone(),
        )
    })
}

/// `buffer`, an output of `problem` or the indices beside it, bound in `element` at the output's
/// lanes.
fn output_bound<M>(
    launcher: &Launcher,
    buffer: &TensorBinding,
    element: ElemType,
    problem: &ReduceRoutineProblem,
    blueprint: &ReduceBlueprint<M>,
) -> Result<Bound, Decline> {
    bound_at(
        launcher,
        problem,
        buffer.clone(),
        SLOTS,
        element,
        problem.box_row_count_at_lane_count(blueprint.lane_count),
    )
}

/// `buffer`, one a launch of `problem` writes holding `along` in place of the reduced axis, bound
/// in `element` at `lane_count` ([`ReduceRoutineProblem::labelled`]).
///
/// # Errors
///
/// [`Decline::Broken`] where the launch the launcher was prepared for cannot bind it, which its
/// routine's prepare rules out.
pub(crate) fn bound_at(
    launcher: &Launcher,
    problem: &ReduceRoutineProblem,
    buffer: TensorBinding,
    along: Dimension,
    element: ElemType,
    lane_count: usize,
) -> Result<Bound, Decline> {
    let (buffer, dimensions) = problem.labelled(buffer, along);
    launcher
        .arg(buffer, element)
        .dimensions(&dimensions)
        .vectorize(lane_count)
        .build()
        .or_broken(ADMITTED_PLAN_RUNS)
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
    use crate::kernels::UnitDirectReduceRoutine;
    use cubecl::ir::{FloatKind, IntKind};
    use cubecl::tile::layout::ElemAlignment;
    use cubek_microkernel::reduce::unit::UnitDirectReduceMicrokernel;
    use cubek_std::Hardware;
    use cubek_std::routine::Routine;

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

    /// The bytes of a workspace's buffers that no test here launches with.
    fn workspace() -> ReduceWorkspace {
        let client = cubecl::test_device().client();
        ReduceWorkspace {
            partials: client.empty(64),
            partial_indices: client.empty(32),
            arrivals: client.empty(8),
        }
    }

    #[test]
    fn tensors_given_a_workspace_write_partials_to_it() {
        let tensors = ReduceTensors::zeros(&INPUT_SHAPE, 1, ReduceOperation::Sum);
        let given = tensors.with_workspace(workspace());
        assert_eq!(given.problem().partials, Some(workspace().partials_in(F32)));
    }

    #[test]
    fn an_unbound_input_states_its_own_alignment() {
        let tensors = ReduceTensors::unbound_zeros(&INPUT_SHAPE, 1, ReduceOperation::Sum);
        assert_eq!(
            tensors.stated_problem().elem_alignments[0],
            ElemAlignment::dense_rows(INPUT_SHAPE[1])
        );
    }

    #[test]
    fn tensors_are_shown_by_the_problem_they_state() {
        let tensors = ReduceTensors::zeros(&INPUT_SHAPE, 1, ReduceOperation::Sum);
        let shown = format!("{tensors:?}");
        assert!(
            shown.contains(&format!("{:?}", tensors.problem())),
            "{shown}"
        );
    }

    /// Why a launch the unit routine prepared binds its operands.
    const BINDS: &str = "a routine's launch binds what it prepared";

    /// A sum of rows across the contiguous axis, and what the unit routine prepares for it.
    fn across() -> (
        ReduceTensors,
        ReduceBlueprint<UnitDirectReduceMicrokernel>,
        Grid,
    ) {
        let tensors = ReduceTensors::zeros(&[300, 64], 0, ReduceOperation::Sum);
        let hw = Hardware::from_client(tensors.client());
        let (blueprint, grid) =
            UnitDirectReduceRoutine::prepare_first_served(&hw, tensors.problem())
                .expect("the unit routine serves every problem");
        (tensors, blueprint, grid)
    }

    #[test]
    fn buffers_are_bound_at_the_blueprints_lanes() {
        let (tensors, blueprint, grid) = across();
        let (_, binding) = tensors.bind(&blueprint, &grid).expect(BINDS);
        assert_eq!(
            [binding.input.vector_size, binding.out.vector_size],
            [blueprint.lane_count; 2]
        );
    }

    #[test]
    fn an_output_alone_is_bound_at_the_blueprints_lanes() {
        let (tensors, blueprint, grid) = across();
        let (_, out) = tensors.bind_out(&blueprint, &grid).expect(BINDS);
        assert_eq!(out.vector_size, blueprint.lane_count);
    }

    #[test]
    fn a_binding_is_made_into_what_its_operands_are() {
        let (tensors, blueprint, grid) = across();
        let lanes = |operand: Bound| operand.vector_size;
        let (_, binding) = tensors.bind(&blueprint, &grid).expect(BINDS);
        let made = bound_as(binding, lanes, lanes);
        assert_eq!([made.input, made.out], [blueprint.lane_count; 2]);
    }

    /// A buffer of the output's shape bound at one lane, whatever the output's are.
    #[test]
    fn a_buffer_is_bound_at_the_lanes_it_is_given() {
        let (tensors, blueprint, grid) = across();
        let (launcher, _) = tensors.bind(&blueprint, &grid).expect(BINDS);
        let bound = bound_at(
            &launcher,
            tensors.problem(),
            tensors.out.clone(),
            SLOTS,
            F32,
            1,
        )
        .expect(BINDS);
        assert_eq!(bound.vector_size, 1);
    }

    #[test]
    fn an_unbound_input_is_handed_its_geometry() {
        let tensors = ReduceTensors::unbound_zeros(&INPUT_SHAPE, 1, ReduceOperation::Sum);
        let hw = Hardware::from_client(tensors.client());
        let (blueprint, grid) =
            UnitDirectReduceRoutine::prepare_first_served(&hw, tensors.problem())
                .expect("the unit routine serves every problem");
        let (_, binding) = tensors.bind(&blueprint, &grid).expect(BINDS);
        assert_eq!(binding.input.geometry.shape(), INPUT_SHAPE);
    }
}
