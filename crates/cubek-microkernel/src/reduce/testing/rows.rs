use crate::reduce::{BATCH, REDUCED, ReduceOperation};
use cubecl::tile::Dimension;

/// Which way a case's lines run against its rows: the buffers a case lays out, and the
/// dimension its work item's lines run along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LineDirection {
    /// Along the reduced dimension, a line `width` values of one row.
    Along,
    /// Across it, a line one value of `width` rows side by side.
    Across,
}

impl LineDirection {
    /// The direction of lines that run along `lines_along`.
    pub fn from_lines_along(lines_along: Dimension) -> Self {
        match lines_along == REDUCED {
            true => Self::Along,
            false => Self::Across,
        }
    }

    /// The dimension lines in this direction run along.
    pub fn to_lines_along(self) -> Dimension {
        match self {
            Self::Along => REDUCED,
            Self::Across => BATCH,
        }
    }
}

/// What a case's rows hold, each kind a behavior of its own.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Rows {
    /// Small whole numbers, so every sum is exact, on a cycle of eleven that row `r` starts `r`
    /// values in and raises by `r`, so no two rows agree; for a product, `±1`, and `2` and `0.5`
    /// in turn on a cycle of nine, so no partial product of a long row overflows whatever order it
    /// multiplies in.
    Plain,
    /// Plain rows whose largest value sits at positions 1 and `len - 1` and smallest at 2 and
    /// `len - 2`: the lower index must win the tie.
    Ties,
    /// Plain rows with a NaN halfway and at the end.
    Nan,
    /// Every value `-inf`.
    NegativeInfinity,
    /// Every value zero.
    Zeros,
}

/// The value a tie sits at, past every plain value.
const TIE: f32 = 1000.0;

impl Rows {
    /// Row `r` of `len` values, as `operation` reads it.
    pub(crate) fn row(self, operation: ReduceOperation, len: usize, r: usize) -> Vec<f32> {
        match self {
            Rows::Plain => plain_row(operation, len, r),
            Rows::Ties => with_values(plain_row(operation, len, r), &ties_of_len(len)),
            Rows::Nan => with_values(plain_row(operation, len, r), &nans_of_len(len)),
            Rows::NegativeInfinity => vec![f32::NEG_INFINITY; len],
            Rows::Zeros => vec![0.0; len],
        }
    }
}

/// Row `r` of [`Rows::Plain`], `len` values long, as `operation` reads it.
fn plain_row(operation: ReduceOperation, len: usize, r: usize) -> Vec<f32> {
    match operation {
        ReduceOperation::Prod => (r..r + len).map(product_value_of_index).collect(),
        _ => (r..r + len)
            .map(|i| ((i * 7) % 11) as f32 - 5.0 + r as f32)
            .collect(),
    }
}

/// Value `i` of the cycle a product's rows hold.
fn product_value_of_index(i: usize) -> f32 {
    match i % 9 {
        0 => 2.0,
        4 => 0.5,
        3 | 5 => -1.0,
        _ => 1.0,
    }
}

/// The positions and values of the ties of a row of `len` values.
fn ties_of_len(len: usize) -> [(usize, f32); 4] {
    [(1, TIE), (len - 1, TIE), (2, -TIE), (len - 2, -TIE)]
}

/// The positions and values of the NaN of a row of `len` values.
fn nans_of_len(len: usize) -> [(usize, f32); 2] {
    [len / 2, len - 1].map(|position| (position, f32::NAN))
}

/// `row` with each `(position, value)` of `at` written in.
fn with_values(mut row: Vec<f32>, at: &[(usize, f32)]) -> Vec<f32> {
    for &(position, value) in at {
        row[position] = value;
    }
    row
}
