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
    /// How many values a row holds, at its bucket ([`bucket_of_reduced_values`](Self::bucket_of_reduced_values)).
    reduced: usize,
    /// How many rows there are, at their bucket ([`bucket_of_rows`](Self::bucket_of_rows)).
    rows: usize,
    /// Whether the caller gave scratch: a call given none cannot run a winner that takes it.
    scratch: bool,
}

impl ReduceKey {
    /// The extent a key holds for rows of `values` values: the power of four at or above them,
    /// from 16 to 2^20.
    fn bucket_of_reduced_values(values: usize) -> usize {
        power_at_or_above(4, values).clamp(16, LARGEST_BUCKET)
    }

    /// The extent a key holds for `rows` rows: the power of eight at or above them, at most 2^20.
    fn bucket_of_rows(rows: usize) -> usize {
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
fn widest_line_within_row_of_values(values: usize) -> usize {
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
        let input_line =
            widest_line_of(input).min(widest_line_within_row_of_values(problem.line_stays_in()));
        Self::new(
            problem.elements,
            problem.indices.map(|indices| indices.element),
            problem.operation,
            problem.lines_along(),
            input_line,
            out_line,
            Self::bucket_of_reduced_values(problem.reduced_extent()),
            Self::bucket_of_rows(problem.row_count()),
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
    use cubecl::std::tensor::TensorHandle;

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
        assert_eq!(ReduceKey::bucket_of_reduced_values(4097), 16384);
    }

    #[test]
    fn a_row_length_a_power_of_four_is_its_own_bucket() {
        assert_eq!(ReduceKey::bucket_of_reduced_values(4096), 4096);
    }

    /// A row count's bucket is the power of eight at or above it.
    #[test]
    fn a_row_count_is_keyed_by_a_power_of_eight() {
        assert_eq!(ReduceKey::bucket_of_rows(65), 512);
    }

    /// Past the largest bucket, every row count shares it.
    #[test]
    fn a_row_count_past_the_largest_bucket_is_keyed_by_it() {
        assert_eq!(
            ReduceKey::bucket_of_rows(LARGEST_BUCKET + 1),
            LARGEST_BUCKET
        );
    }

    /// Its winner may be one that takes the scratch.
    #[test]
    fn a_call_given_scratch_is_keyed_by_it() {
        let problem = ReduceRoutineProblem {
            workspace: ReduceWorkspaceBytes::UNBOUNDED,
            ..ReduceRoutineProblem::dense(&[4, 300], 1, ReduceOperation::Sum)
        };
        assert!(ReduceKey::from(&problem).scratch);
    }

    #[test]
    fn a_call_given_no_scratch_is_keyed_by_none() {
        let problem = ReduceRoutineProblem::dense(&[4, 300], 1, ReduceOperation::Sum);
        assert!(!ReduceKey::from(&problem).scratch);
    }

    const REDUCED_BUCKET: usize = 1024;
    const ROWS_BUCKET: usize = 64;
    const LINE: usize = 4;

    /// The key of sums in lines of [`LINE`] that lie along `lines_along`.
    fn key_of_lines(lines_along: Dimension) -> ReduceKey {
        ReduceKey {
            elements: ReduceElements {
                input: F32,
                accumulator: F32,
                out: F32,
            },
            indices: None,
            operation: ReduceOperation::Sum,
            lines_along,
            input_line: LINE,
            out_line: LINE,
            reduced: REDUCED_BUCKET,
            rows: ROWS_BUCKET,
            scratch: false,
        }
    }

    /// The problem [`key_of_lines`] stands for: `shape`, reduced along `axis`.
    fn problem_in_lines(shape: &[usize], axis: usize) -> ReduceRoutineProblem {
        ReduceRoutineProblem {
            elem_alignments: [ElemAlignment::dense_rows(LINE); 2],
            ..ReduceRoutineProblem::dense(shape, axis, ReduceOperation::Sum)
        }
    }

    #[test]
    fn a_key_along_the_axis_stands_for_rows_of_its_buckets() {
        assert_eq!(
            ReduceRoutineProblem::from(&key_of_lines(REDUCED)),
            problem_in_lines(&[ROWS_BUCKET, REDUCED_BUCKET], 1)
        );
    }

    #[test]
    fn a_key_across_the_axis_stands_for_columns_of_its_buckets() {
        assert_eq!(
            ReduceRoutineProblem::from(&key_of_lines(BATCH)),
            problem_in_lines(&[REDUCED_BUCKET, ROWS_BUCKET], 0)
        );
    }

    #[test]
    fn a_keys_problem_along_the_axis_has_that_key() {
        let problem = problem_in_lines(&[ROWS_BUCKET, REDUCED_BUCKET], 1);
        assert_eq!(ReduceKey::from(&problem), key_of_lines(REDUCED));
    }

    #[test]
    fn a_keys_problem_across_the_axis_has_that_key() {
        let problem = problem_in_lines(&[REDUCED_BUCKET, ROWS_BUCKET], 0);
        assert_eq!(ReduceKey::from(&problem), key_of_lines(BATCH));
    }

    /// A fused reduction finds the candidate its unfused round settled: its key holds the line its
    /// input's own layout takes, not the two values the caller binds.
    #[test]
    fn an_unbound_call_keys_as_its_bound_call() {
        let client = cubecl::test_device().client();
        let shape = [ROWS_BUCKET, REDUCED_BUCKET];
        let out = TensorHandle::empty(&client, [ROWS_BUCKET, 1], F32);
        let key = key_of_lines(REDUCED);
        let tensors = ReduceTensors::unbound(
            &client,
            Geometry::row_major(&shape),
            2,
            out.binding(),
            1,
            key.operation,
            key.elements,
        );
        assert_eq!(ReduceKey::from(&tensors).input_line, WIDEST_KEYED_LINE);
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

    #[test]
    fn a_stored_piece_takes_its_own_width() {
        const PIECE: usize = 8;
        assert_eq!(widest_line_of(ElemAlignment::stored_piece(PIECE)), PIECE);
    }
}
