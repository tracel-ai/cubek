//! How a reduce launch reads its operands and binds them.

use crate::kernels::{REDUCED, ReduceBlueprint, ReduceElements, ReduceRoutineProblem, SLOTS};
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::TensorBinding;
use cubecl::tile::launch::{Bound, Decline, Unbound};
use cubecl::tile::{Dimension, Geometry, Launcher};
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

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
}

impl ReduceTensors<TensorBinding> {
    /// `out = operation(input)` along axis `axis`: `out` has `input`'s shape, `axis` one wide, or
    /// `k` wide for a top-k.
    ///
    /// # Panics
    ///
    /// Where `out`'s shape is not that.
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
            problem,
        }
    }

    /// `values = operation(input)` along axis `axis`, and the indices of those values to
    /// `indices`, in `index_element`, from one launch: `indices` has `values`' shape, each written
    /// at its own strides.
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and where the operation's values carry no index (a sum's do not),
    /// or `indices` does not have `values`' shape.
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
            "{operation:?} accumulates the values themselves, which carry no index"
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

    /// Both operands bound as `blueprint` states them, on `grid`.
    ///
    /// # Panics
    ///
    /// Where `blueprint` and `grid` were not prepared for these tensors' problem.
    pub(crate) fn bind<M>(&self, blueprint: &ReduceBlueprint<M>, grid: &Grid) -> ReduceBinding {
        ReduceBinding::bound(
            &self.client,
            &self.problem,
            [&self.input, &self.out],
            self.indices.as_ref(),
            blueprint,
            grid,
        )
    }
}

impl ReduceTensors<Geometry> {
    /// `out = operation(input)` along dim `axis`, the input laid out as `input` but bound by the
    /// caller once the launch is prepared and read at exactly `width`: what a fused read stands
    /// for, whose lines the caller's code computes whole.
    ///
    /// # Panics
    ///
    /// As [`new`](ReduceTensors::new).
    pub fn unbound(
        client: &Client,
        input: Geometry,
        width: usize,
        out: TensorBinding,
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
    ) -> Self {
        let problem =
            ReduceRoutineProblem::of_unbound(&input, width, &out, axis, operation, elements);
        Self {
            client: client.clone(),
            input,
            out,
            indices: None,
            problem,
        }
    }

    /// The input's spec and geometry as `blueprint` states them, on `grid`, for the caller to
    /// bind, and the output bound.
    ///
    /// # Panics
    ///
    /// Where `blueprint` and `grid` were not prepared for these tensors' problem.
    pub(crate) fn bind<M>(
        &self,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> ReduceBinding<Unbound, Bound> {
        ReduceBinding::unbound(
            &self.client,
            &self.problem,
            &self.input,
            &self.out,
            blueprint,
            grid,
        )
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
        }
    }
}

impl ReduceBinding {
    /// `input` and `out`, and `indices` beside them where there are, bound as `blueprint`
    /// states them for `problem`, on `grid`.
    fn bound<M>(
        client: &Client,
        problem: &ReduceRoutineProblem,
        [input, out]: [&TensorBinding; 2],
        indices: Option<&TensorBinding>,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Self {
        let launcher = launcher_of(client, problem, blueprint, grid);
        let (input, dimensions) = problem.labelled(input.clone(), REDUCED);
        let input = launcher
            .arg(input, problem.elements.input)
            .dimensions(&dimensions)
            .boundary(problem.input_boundary())
            .vectorize(blueprint.width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let out = out_bound(&launcher, out, problem.elements.out, problem, blueprint);
        let indices = indices.map(|indices| {
            let element = problem
                .indices
                .expect("indices beside the values have an element")
                .element;
            out_bound(&launcher, indices, element, problem, blueprint)
        });
        ReduceBinding {
            launcher,
            input,
            out,
            indices,
        }
    }
}

impl ReduceBinding<Unbound, Bound> {
    /// The spec and geometry of `input`, which the caller binds, as `blueprint` states them for
    /// `problem`, on `grid`, and `out` bound.
    fn unbound<M>(
        client: &Client,
        problem: &ReduceRoutineProblem,
        input: &Geometry,
        out: &TensorBinding,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> Self {
        let launcher = launcher_of(client, problem, blueprint, grid);
        let (input, dimensions) = problem.labelled_geometry(input, REDUCED);
        let input = launcher
            .unbound(&input)
            .dimensions(&dimensions)
            .boundary(problem.input_boundary())
            .vectorize(blueprint.width)
            .build_spec()
            .expect(ADMITTED_PLAN_RUNS);
        let out = out_bound(&launcher, out, problem.elements.out, problem, blueprint);
        ReduceBinding {
            launcher,
            input,
            out,
            indices: None,
        }
    }
}

/// The launcher `grid` builds for `problem` placed by `blueprint`.
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
/// width.
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
        blueprint.out_width,
    )
}

/// `buffer`, one a launch of `problem` writes holding `along` in place of the reduced axis, bound
/// in `element` at `width` ([`ReduceRoutineProblem::labelled`]).
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
    width: usize,
) -> Bound {
    let (buffer, dimensions) = problem.labelled(buffer, along);
    launcher
        .arg(buffer, element)
        .dimensions(&dimensions)
        .vectorize(width)
        .build()
        .expect(ADMITTED_PLAN_RUNS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{FloatKind, IntKind};
    use cubecl::zspace::Shape;
    use cubek_test_utils::TestInput;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A buffer of `shape` in f32 on the test device, holding zeros.
    fn zeros(client: &Client, shape: &[usize]) -> TensorBinding {
        TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
            .dtype(F32)
            .zeros()
            .generate_without_host_data()
            .binding()
    }

    /// `ReduceTensors::with_indices` for `operation` along the last axis of a `[4, 8]` input, the
    /// values one wide and the indices `indices_shape`.
    fn tensors_with_indices(operation: ReduceOperation, indices_shape: &[usize]) -> ReduceTensors {
        let client = cubecl::test_device().client();
        ReduceTensors::with_indices(
            &client,
            zeros(&client, &[4, 8]),
            zeros(&client, &[4, 1]),
            zeros(&client, indices_shape),
            1,
            operation,
            ReduceElements {
                input: F32,
                accumulator: F32,
                out: F32,
            },
            ElemType::Int(IntKind::I32),
        )
    }

    /// A sum's values carry no index: with_indices' documented panic.
    #[test]
    #[should_panic]
    fn indices_beside_a_sum_panic() {
        tensors_with_indices(ReduceOperation::Sum, &[4, 1]);
    }

    /// Indices of another shape than the values: with_indices' documented panic.
    #[test]
    #[should_panic]
    fn indices_of_another_shape_panic() {
        tensors_with_indices(ReduceOperation::ArgMax, &[1, 4]);
    }
}
