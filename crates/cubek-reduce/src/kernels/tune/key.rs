//! What tells one reduction from another for the autotune ([`ReduceKey`]).

use crate::kernels::{
    ReduceElements, ReduceIndices, ReduceRoutineProblem, ReduceTensors, ReduceWorkspaceBytes,
};
use cubecl::AutotuneKey;
use cubecl::ir::ElemType;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Dimension, Geometry};
use cubek_microkernel::reduce::{REDUCED, ReduceOperation};
use serde::{Deserialize, Serialize};

/// What tells one reduction from another for the autotune: its elements, its operation, how its
/// lines lie, the widest line each buffer takes, how long a row is and how many rows there are.
/// How many axes the rows run along is not: every routine places its owners along one dimension
/// of rows whatever the rank. Every reduce routine prepares its Blueprint from the problem a key
/// stands for ([`ReduceRoutineProblem::keyed`]), so the widths a key holds are ones every call
/// of it reads at.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct ReduceKey {
    elements: ReduceElements,
    indices: Option<ElemType>,
    operation: ReduceOperation,
    lines_along: Dimension,
    /// The widest line, a power of two, the input's buffer takes.
    input_line: usize,
    /// The widest line, a power of two, the output's buffer takes, and the indices' beside it.
    out_line: usize,
    #[autotune(anchor(exp(min = 16, max = 1048576, base = 4)))]
    reduced: usize,
    #[autotune(anchor(exp(max = 1048576, base = 8)))]
    rows: usize,
}

/// The widest line a key tells apart: past every device's vector width.
const WIDEST_KEYED_LINE: usize = 64;

/// The widest line, a power of two at most [`WIDEST_KEYED_LINE`], that `alignment` takes.
fn widest_line_of(alignment: ElemAlignment) -> usize {
    let mut line = WIDEST_KEYED_LINE;
    while !alignment.admits(line) {
        line /= 2;
    }
    line
}

impl From<&ReduceTensors> for ReduceKey {
    fn from(tensors: &ReduceTensors) -> Self {
        Self::from(tensors.problem())
    }
}

impl From<&ReduceTensors<Geometry>> for ReduceKey {
    /// The key of the call whose input the caller binds, read off the input's own geometry
    /// rather than the width the caller binds it at: the key a call binding that input gets, so a
    /// fused reduction finds the candidate its unfused round settled.
    fn from(tensors: &ReduceTensors<Geometry>) -> Self {
        Self::from(&tensors.stated_problem())
    }
}

impl From<&ReduceRoutineProblem> for ReduceKey {
    fn from(problem: &ReduceRoutineProblem) -> Self {
        let [input, out] = problem.elem_alignments;
        let out_line = problem.indices.map_or(widest_line_of(out), |indices| {
            widest_line_of(out).min(widest_line_of(indices.elem_alignment))
        });
        Self::new(
            problem.elements,
            problem.indices.map(|indices| indices.element),
            problem.operation,
            problem.lines_along(),
            widest_line_of(input),
            out_line,
            problem.reduced_extent(),
            problem.row_count(),
        )
    }
}

impl From<&ReduceKey> for ReduceRoutineProblem {
    /// A matrix of `rows` rows of `reduced` values: the reduced axis the inner one where the lines
    /// lie along it, the outer where they lie across it, each buffer aligned to the widest line
    /// the key holds for it.
    fn from(key: &ReduceKey) -> Self {
        let (shape, axis) = match key.lines_along == REDUCED {
            true => (vec![key.rows, key.reduced], 1),
            false => (vec![key.reduced, key.rows], 0),
        };
        let [input_alignment, out_alignment] =
            [key.input_line, key.out_line].map(ElemAlignment::dense_rows);
        Self {
            elem_alignments: [input_alignment, out_alignment],
            contiguous: Some(shape.len() - 1),
            shape,
            axis,
            operation: key.operation,
            elements: key.elements,
            indices: key.indices.map(|element| ReduceIndices {
                element,
                elem_alignment: out_alignment,
            }),
            // It stands for calls whose caller gave the scratch they need.
            workspace: ReduceWorkspaceBytes::UNBOUNDED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::BATCH;

    /// The problem a key stands for has that key, for rows along the reduced dim.
    #[test]
    fn a_keys_problem_along_the_axis_has_that_key() {
        let key = ReduceKey::from(&ReduceRoutineProblem::dense(
            &[3, 7, 300],
            2,
            ReduceOperation::Sum,
        ));
        assert_eq!(ReduceKey::from(&ReduceRoutineProblem::from(&key)), key);
    }

    /// The problem a key stands for has that key, for rows across the reduced dim.
    #[test]
    fn a_keys_problem_across_the_axis_has_that_key() {
        let key = ReduceKey::from(&ReduceRoutineProblem::dense(
            &[3, 300, 7],
            1,
            ReduceOperation::Sum,
        ));
        assert_eq!(ReduceKey::from(&ReduceRoutineProblem::from(&key)), key);
    }

    /// A key's problem lays its lines as the key says.
    #[test]
    fn a_keys_problem_lays_its_lines_as_the_key_says() {
        let key = ReduceKey::from(&ReduceRoutineProblem::dense(
            &[4, 300, 7],
            1,
            ReduceOperation::Sum,
        ));
        assert_eq!(ReduceRoutineProblem::from(&key).lines_along(), BATCH);
    }

    /// A key holds the widest line its input takes: rows 300 long take lines of 4.
    #[test]
    fn a_key_holds_the_widest_line_its_input_takes() {
        const ROW_LEN: usize = 300;
        let key = ReduceKey::from(&ReduceRoutineProblem::dense(
            &[8, ROW_LEN],
            1,
            ReduceOperation::Sum,
        ));
        assert_eq!(key.input_line, 4);
    }

    /// A buffer read in stored pieces takes lines of its piece alone.
    #[test]
    fn a_stored_piece_takes_its_own_width() {
        const PIECE: usize = 8;
        assert_eq!(widest_line_of(ElemAlignment::stored_piece(PIECE)), PIECE);
    }
}
