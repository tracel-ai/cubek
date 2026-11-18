//! How a reduce launch reads its operands and binds them.

use crate::kernels::{REDUCED, ReduceBlueprint, ReduceElements, ReduceRoutineProblem, TOP};
use cubecl::client::Client;
use cubecl::prelude::TensorBinding;
use cubecl::tile::launch::{Bound, Refusal, Unbound};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Geometry, Launcher};
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// Why an unbound reduction declines where its microkernel reads narrower lines than the caller's
/// input is stated at ([`ReduceTensors::unbound`]): the caller computes its lines whole.
pub const UNBOUND_LINES_NARROWER: Refusal = Refusal::Declined {
    why: "the microkernel reads narrower lines than the caller's input is stated at",
};

/// The input and the output of one reduction, and the problem they state: the input a buffer
/// ([`TensorBinding`]) or the geometry of an input the caller binds once the launch is prepared
/// ([`Geometry`], [`ReduceLauncher::launch_unbound`](crate::kernels::ReduceLauncher::launch_unbound)),
/// the output a buffer.
///
/// Both are read where they lie, at their own strides: every axis of each is a dimension of the
/// problem, so a reduction along any axis of a tensor of any rank binds without a copy.
#[derive(Clone)]
pub struct ReduceTensors<In = TensorBinding> {
    client: Client,
    input: In,
    out: TensorBinding,
    /// The indices written beside the values, laid out as `out`, where they are.
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
        let problem = ReduceRoutineProblem::stated(
            &Geometry::from(&input),
            &Geometry::from(&out),
            axis,
            operation,
            elements,
        );
        Self {
            client: client.clone(),
            input,
            out,
            indices: None,
            problem,
        }
    }

    /// `values = operation(input)` along axis `axis`, and the indices of those values to
    /// `indices`, in `index_element`, from one launch: `indices` has `values`' shape and strides.
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and where the operation's values carry no index (a sum's do not),
    /// or `indices` is not laid out as `values`.
    #[allow(clippy::too_many_arguments)]
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
            indices.shape == values.shape && indices.strides == values.strides,
            "the indices are laid out as the values"
        );
        let mut tensors = Self::new(client, input, values, axis, operation, elements);
        tensors.problem.indices = Some(index_element);
        tensors.indices = Some(indices);
        tensors
    }

    /// Both operands bound as `blueprint` states them, on `grid`.
    pub fn bind<M>(&self, blueprint: &ReduceBlueprint<M>, grid: &Grid) -> ReduceBinding {
        let launcher = grid.launcher(&self.client, || {
            (
                self.problem.kernel(&blueprint.partitioning),
                blueprint.partitioning.space().clone(),
            )
        });
        let (input, input_labels) = self.problem.fold(self.input.clone(), REDUCED);
        let input = launcher
            .arg(input, self.problem.elements.input)
            .dimensions(&input_labels)
            .boundary(self.problem.input_boundary())
            .vectorize(blueprint.width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let (out, out_labels) = self.problem.fold(self.out.clone(), TOP);
        let out = launcher
            .arg(out, self.problem.elements.out)
            .dimensions(&out_labels)
            .vectorize(blueprint.out_width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let indices = self.indices.as_ref().map(|indices| {
            let (indices, labels) = self.problem.fold(indices.clone(), TOP);
            launcher
                .arg(
                    indices,
                    self.problem
                        .indices
                        .expect("indices beside the values have an element"),
                )
                .dimensions(&labels)
                .vectorize(blueprint.out_width)
                .build()
                .expect(ADMITTED_PLAN_RUNS)
        });
        ReduceBinding {
            launcher,
            input,
            out,
            indices,
        }
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
        let mut problem =
            ReduceRoutineProblem::stated(&input, &Geometry::from(&out), axis, operation, elements);
        problem.elem_alignments[0] = ElemAlignment::stored_piece(width);
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
    pub fn bind<M>(
        &self,
        blueprint: &ReduceBlueprint<M>,
        grid: &Grid,
    ) -> ReduceBinding<Unbound, Bound> {
        let launcher = grid.launcher(&self.client, || {
            (
                self.problem.kernel(&blueprint.partitioning),
                blueprint.partitioning.space().clone(),
            )
        });
        let (input, input_labels) = self.problem.fold_geometry(&self.input, REDUCED);
        let input = launcher
            .unbound(&input)
            .dimensions(&input_labels)
            .boundary(self.problem.input_boundary())
            .vectorize(blueprint.width)
            .build_spec()
            .expect(ADMITTED_PLAN_RUNS);
        let (out, out_labels) = self.problem.fold(self.out.clone(), TOP);
        let out = launcher
            .arg(out, self.problem.elements.out)
            .dimensions(&out_labels)
            .vectorize(blueprint.out_width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        ReduceBinding {
            launcher,
            input,
            out,
            indices: None,
        }
    }
}

/// The operands of one launch and the launcher they were bound through: each a [`Bound`] buffer,
/// an [`Unbound`] spec the caller binds, or the launch argument either became.
pub struct ReduceBinding<In = Bound, Out = In> {
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
