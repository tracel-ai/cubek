//! How a reduce launch reads its operands and binds them.

use crate::kernels::{ReduceElements, ReduceRoutineBlueprint, ReduceRoutineProblem};
use cubecl::client::Client;
use cubecl::prelude::TensorBinding;
use cubecl::tile::Launcher;
use cubecl::tile::launch::Bound;
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::reduce::ReduceOperation;
use cubek_std::routine::{ADMITTED_PLAN_RUNS, Grid};

/// The input and the output of one reduction, and the problem they state.
///
/// Both are read where they lie, at their own strides: every dim of each is an axis of the
/// problem, so a reduction along any dim of a tensor of any rank binds without a copy.
#[derive(Clone)]
pub struct ReduceTensors {
    client: Client,
    input: TensorBinding,
    out: TensorBinding,
    problem: ReduceRoutineProblem,
}

impl core::fmt::Debug for ReduceTensors {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReduceTensors")
            .field("problem", &self.problem)
            .finish_non_exhaustive()
    }
}

impl ReduceTensors {
    /// `out = operation(input)` along dim `axis`: `out` has `input`'s shape, `axis` one wide, or
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
        let shape = input.shape.to_vec();
        let expected: Vec<usize> = shape
            .iter()
            .enumerate()
            .map(|(d, extent)| match d == axis {
                true => operation.written_per_row(),
                false => *extent,
            })
            .collect();
        assert_eq!(
            out.shape.to_vec(),
            expected,
            "a reduction along dim {axis} writes the input's shape with that dim {} wide",
            operation.written_per_row()
        );
        // A line runs along the innermost dim where both operands step one element along it.
        let inner = shape.len() - 1;
        let steps_one =
            |binding: &TensorBinding| binding.strides[inner] == 1 || binding.shape[inner] == 1;
        let contiguous = (steps_one(&input) && steps_one(&out)).then_some(inner);
        let problem = ReduceRoutineProblem {
            elem_alignments: [&input, &out]
                .map(|binding| ElemAlignment::new(&binding.shape, &binding.strides)),
            shape,
            axis,
            operation,
            elements,
            contiguous,
        };
        Self {
            client: client.clone(),
            input,
            out,
            problem,
        }
    }

    pub fn problem(&self) -> &ReduceRoutineProblem {
        &self.problem
    }

    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Both operands bound as `blueprint` states them, on `grid`.
    pub fn bind<M>(&self, blueprint: &ReduceRoutineBlueprint<M>, grid: &Grid) -> ReduceBinding {
        let launcher = grid.launcher(&self.client, || {
            (
                blueprint.partitioning.clone(),
                blueprint.partitioning.space().clone(),
            )
        });
        let input = launcher
            .arg(self.input.clone(), self.problem.elements.input)
            .axes(&self.problem.input_axes())
            .vectorize(blueprint.width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        let out = launcher
            .arg(self.out.clone(), self.problem.elements.out)
            .axes(&self.problem.out_axes())
            .vectorize(blueprint.out_width)
            .build()
            .expect(ADMITTED_PLAN_RUNS);
        ReduceBinding {
            launcher,
            input,
            out,
        }
    }
}

/// The operands of one launch, bound, and the launcher they were bound through.
pub struct ReduceBinding {
    pub launcher: Launcher,
    pub input: Bound,
    pub out: Bound,
}
