//! What tells one reduction from another for the autotune ([`ReduceKey`]).

use crate::kernels::{ReduceElements, ReduceIndices, ReduceRoutineProblem, ReduceTensors};
use cubecl::AutotuneKey;
use cubecl::ir::ElemType;
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::reduce::{ReduceLayout, ReduceOperation};
use serde::{Deserialize, Serialize};

/// What tells one reduction from another for the autotune: its elements, its operation, how its
/// lines lie, how many dims its rows run along, how long a row is and how many rows there are.
///
/// The row dims are kept whole, since the cube and the hypercube decline more than they place
/// their cubes along: a winner for a matrix would otherwise be launched on a tensor of more dims.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ReduceKey {
    elements: ReduceElements,
    indices: Option<ElemType>,
    operation: ReduceOperation,
    layout: ReduceLayout,
    row_dims: usize,
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
            problem.layout(),
            problem.rows().len(),
            problem.reduced_extent(),
            problem.row_count(),
        )
    }
}

impl From<&ReduceKey> for ReduceRoutineProblem {
    /// A dense tensor of `rows` rows of `reduced` values, its rows along the innermost of its row
    /// dims, the others one wide: the reduced dim the innermost where the lines lie along it, and
    /// next to it where they lie across it.
    fn from(key: &ReduceKey) -> Self {
        let (shape, axis) = match key.row_dims {
            0 => (vec![key.reduced], 0),
            _ => shape_with_rows(key),
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

/// The shape of the dense tensor `key` describes, whose rows run along at least one dim, and its
/// reduced axis: the innermost where the lines lie along it, next to it where they lie across it.
fn shape_with_rows(key: &ReduceKey) -> (Vec<usize>, usize) {
    let outer = vec![1; key.row_dims - 1];
    match key.layout {
        ReduceLayout::Along => (
            [outer.as_slice(), &[key.rows, key.reduced]].concat(),
            key.row_dims,
        ),
        ReduceLayout::Across => (
            [outer.as_slice(), &[key.reduced, key.rows]].concat(),
            key.row_dims - 1,
        ),
    }
}
