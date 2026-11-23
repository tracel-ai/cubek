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
/// lines lie, the widest line each buffer takes, how long a row is, how many rows there are and
/// whether its caller gave scratch. How many axes the rows run along is not: every routine places
/// its owners along one dimension of rows whatever the rank.
///
/// **The rows are bucketed in powers of two, whatever the autotune level.** cubecl's anchor scales
/// its base by the level, so a bucket would be a power of three or five, which no line of a key
/// divides: the problem a key stands for would be read a value a line, and the candidates ranked
/// at a width its calls do not run.
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
    /// How many values a row holds, at its bucket ([`reduced_bucket`](Self::reduced_bucket)).
    reduced: usize,
    /// How many rows there are, at their bucket ([`rows_bucket`](Self::rows_bucket)).
    rows: usize,
    /// Whether the caller gave scratch: a call given none cannot run a winner that takes it.
    scratch: bool,
}

impl ReduceKey {
    /// The extent a key holds for rows of `values` values: the power of four at or above them,
    /// from 16 to 2^20.
    fn reduced_bucket(values: usize) -> usize {
        power_at_or_above(4, values).clamp(16, LARGEST_BUCKET)
    }

    /// The extent a key holds for `rows` rows: the power of eight at or above them, at most 2^20.
    fn rows_bucket(rows: usize) -> usize {
        power_at_or_above(8, rows).min(LARGEST_BUCKET)
    }
}

/// The extent past which every call shares a bucket.
const LARGEST_BUCKET: usize = 1 << 20;

/// The power of `base` at or above `extent`, one for none.
fn power_at_or_above(base: usize, extent: usize) -> usize {
    let below = base.pow(extent.max(1).ilog(base));
    match below < extent {
        true => below * base,
        false => below,
    }
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

/// The widest line, a power of two at most [`WIDEST_KEYED_LINE`], that divides `values`: the
/// widest that keeps every line within one row `values` long.
fn widest_line_within(values: usize) -> usize {
    match values {
        0 => WIDEST_KEYED_LINE,
        values => (1 << values.trailing_zeros()).min(WIDEST_KEYED_LINE),
    }
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
        // A line runs within its axis, whatever the buffer's alignment says.
        let input_line = widest_line_of(input).min(widest_line_within(problem.line_stays_in()));
        Self::new(
            problem.elements,
            problem.indices.map(|indices| indices.element),
            problem.operation,
            problem.lines_along(),
            input_line,
            out_line,
            Self::reduced_bucket(problem.reduced_extent()),
            Self::rows_bucket(problem.row_count()),
            problem.workspace != ReduceWorkspaceBytes::default(),
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
            workspace: match key.scratch {
                true => ReduceWorkspaceBytes::UNBOUNDED,
                false => ReduceWorkspaceBytes::default(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::BATCH;
    use cubecl::ir::FloatKind;

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// Rows one value long take lines one wide along them, though the buffer is one contiguous
    /// run of every row.
    #[test]
    fn rows_one_value_long_take_lines_one_wide() {
        let input = Geometry::row_major(&[128, 1, 1]);
        let problem = ReduceRoutineProblem::stated(
            &input,
            &input,
            2,
            ReduceOperation::Sum,
            ReduceElements::for_operation(ReduceOperation::Sum, F32, None),
        );
        assert_eq!(ReduceKey::from(&problem).input_line, 1);
    }

    /// A row length's bucket is the power of four at or above it.
    #[test]
    fn a_row_length_is_keyed_by_a_power_of_four() {
        assert_eq!(ReduceKey::reduced_bucket(4097), 16384);
    }

    /// A power of four is its own bucket.
    #[test]
    fn a_row_length_a_power_of_four_is_its_own_bucket() {
        assert_eq!(ReduceKey::reduced_bucket(4096), 4096);
    }

    /// A row count's bucket is the power of eight at or above it.
    #[test]
    fn a_row_count_is_keyed_by_a_power_of_eight() {
        assert_eq!(ReduceKey::rows_bucket(65), 512);
    }

    /// Past the largest bucket, every row count shares it.
    #[test]
    fn a_row_count_past_the_largest_bucket_is_keyed_by_it() {
        assert_eq!(ReduceKey::rows_bucket(LARGEST_BUCKET + 1), LARGEST_BUCKET);
    }

    /// A call given no scratch is keyed apart from one given some, whose winner it may not run.
    #[test]
    fn calls_with_and_without_scratch_are_keyed_apart() {
        let without = ReduceRoutineProblem::dense(&[4, 300], 1, ReduceOperation::Sum);
        let with = ReduceRoutineProblem {
            workspace: ReduceWorkspaceBytes::UNBOUNDED,
            ..without.clone()
        };
        assert_ne!(ReduceKey::from(&without), ReduceKey::from(&with));
    }

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
