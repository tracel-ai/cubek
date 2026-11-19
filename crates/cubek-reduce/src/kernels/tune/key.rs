//! What tells one reduction from another for the autotune ([`ReduceKey`]).

use crate::kernels::{ReduceElements, ReduceIndices, ReduceRoutineProblem, ReduceTensors};
use cubecl::AutotuneKey;
use cubecl::ir::ElemType;
use cubecl::tile::Dimension;
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::reduce::{REDUCED, ReduceOperation};
use serde::{Deserialize, Serialize};

/// What tells one reduction from another for the autotune: its elements, its operation, how its
/// lines lie, how long a row is and how many rows there are. How many axes the rows run along is
/// not: every routine places its owners along one dimension of rows whatever the rank.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ReduceKey {
    elements: ReduceElements,
    indices: Option<ElemType>,
    operation: ReduceOperation,
    lines_along: Dimension,
    #[autotune(anchor(exp(min = 16, max = 1048576, base = 4)))]
    reduced: usize,
    #[autotune(anchor(exp(max = 1048576, base = 8)))]
    rows: usize,
}

impl<In> From<&ReduceTensors<In>> for ReduceKey {
    fn from(tensors: &ReduceTensors<In>) -> Self {
        Self::from(tensors.problem())
    }
}

impl From<&ReduceRoutineProblem> for ReduceKey {
    fn from(problem: &ReduceRoutineProblem) -> Self {
        Self::new(
            problem.elements,
            problem.indices.map(|indices| indices.element),
            problem.operation,
            problem.lines_along(),
            problem.reduced_extent(),
            problem.row_count(),
        )
    }
}

impl From<&ReduceKey> for ReduceRoutineProblem {
    /// A dense matrix of `rows` rows of `reduced` values: the reduced axis the inner one where the
    /// lines lie along it, the outer where they lie across it.
    fn from(key: &ReduceKey) -> Self {
        let (shape, axis) = match key.lines_along == REDUCED {
            true => (vec![key.rows, key.reduced], 1),
            false => (vec![key.reduced, key.rows], 0),
        };
        let mut out_shape = shape.clone();
        out_shape[axis] = key.operation.written_per_row();
        let elem_alignments =
            [&shape, &out_shape].map(|shape| ElemAlignment::dense_rows(shape[shape.len() - 1]));
        let [_, out_alignment] = elem_alignments;
        Self {
            elem_alignments,
            contiguous: Some(shape.len() - 1),
            shape,
            axis,
            operation: key.operation,
            elements: key.elements,
            indices: key.indices.map(|element| ReduceIndices {
                element,
                elem_alignment: out_alignment,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;

    /// The problem a key stands for has the key it came from, along the rows and across them.
    #[test]
    fn a_keys_problem_has_that_key() {
        for (shape, axis) in [([3, 7, 300], 2), ([3, 300, 7], 1)] {
            let key = ReduceKey::from(&ReduceRoutineProblem::dense(
                &shape,
                axis,
                ReduceOperation::Sum,
            ));
            let problem = ReduceRoutineProblem::from(&key);
            assert_eq!(ReduceKey::from(&problem), key, "{shape:?} along {axis}");
        }
    }

    /// A key's problem is dense: its lines lie as the key says, its rows along one dim.
    #[test]
    fn a_keys_problem_is_dense_rows() {
        let key = ReduceKey::from(&ReduceRoutineProblem::dense(
            &[4, 300, 7],
            1,
            ReduceOperation::Sum,
        ));
        let problem = ReduceRoutineProblem::from(&key);
        assert_eq!(problem.lines_along(), crate::kernels::BATCH);
        assert_eq!(problem.contiguous, Some(problem.shape.len() - 1));
    }
}
