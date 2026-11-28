//! Which race a measured product goes to ([`MatmulAutotune`]): **few-rows**, **many-rows** or
//! **block-scaled**. A rule of the selection, not a measurement: the caller that measures builds
//! each race and hands [`select`](super::selector::select) its plan.
//!
//! A product races on a **few-rows** race or a **many-rows** one, whatever its inputs: a routine
//! that does not stage and one that does are different kernel bodies, so one race would be a list
//! whose rows mostly decline, paying a compile each to say so. Within a race a float product and a
//! packed one race the same list, each candidate listed for one kind of product
//! ([`ListedFor`](crate::kernels::ListedFor)) and declining the other before compiling anything.
//!
//! A third race, [`MatmulAutotune::BlockScaled`], multiplies two packed inputs as they are stored
//! on the device's block-scaled instruction of their format: NVFP4's 4-bit by 4-bit, or any other
//! the device has. Naming
//! [`BlockScaledTensorCores`](crate::kernels::MatmulKernel::BlockScaledTensorCores) reaches it at
//! any rows; a measuring strategy sends a product there past the few-rows bound, where both inputs
//! already arrive in a format that instruction reads, so taking it rounds nothing. A decode's rows
//! stay on the few-rows race, bound by the weight's bytes either way.
//!
//! The line is [`FLOAT_FEW_ROWS`] for a float product and [`QUANTIZED_FEW_ROWS`] for a packed
//! one, each a **rule** rather than a measurement: a staged kernel can be competitive on
//! time at a handful of rows while reading a whole input through a tiling nobody needed, so
//! which race a row count belongs to is decided by what the kernels do. A rule tuned to one
//! device's sweep instead is a number every other device has to live with.

use crate::kernels::select::problem::MatmulProblem;

/// The races a measured product goes to, split by the rows a routine that does not stage would
/// stream: each holds the candidates for a float product and for a packed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulAutotune {
    /// A handful of rows, each read once: the routines that do not stage, on packed inputs at the
    /// settings worth trying on them, reading them as stored.
    FewRows,
    /// Everything past the few-rows bound: the staged routines, and on packed inputs the packed
    /// staged routine reading them as stored and the float routines on them decoded first.
    ManyRows,
    /// Two packed inputs multiplied on a GPU's block-scaled tensor cores, at each of
    /// [`BlockScaledTensorCores`](crate::kernels::MatmulKernel::BlockScaledTensorCores)'s
    /// settings: what that pin runs, and nothing else.
    BlockScaled,
}

/// The most rows a float product goes to the few-rows race with: the rows every float few-rows
/// candidate holds whole, each one more accumulator line per cell of its register block (the plane
/// shuffle's `MAX_ROWS` is held to it by a test). Past it, a body that reuses each column's values
/// across rows is the better shape anyway.
pub const FLOAT_FEW_ROWS: usize = 8;

/// Whether a float product of `rows` rows goes to the few-rows race: from one row to
/// [`FLOAT_FEW_ROWS`]. A float product of no rows goes to the many-rows race, which has a
/// candidate for it.
pub fn float_few_rows(rows: usize) -> bool {
    (1..=FLOAT_FEW_ROWS).contains(&rows)
}

/// The most rows a packed product goes to the few-rows race with. Past it the staged routines
/// win on arithmetic, and measuring a routine that does not stage at prefill widths would cost
/// minutes of first-touch tuning for a candidate that cannot win there.
pub const QUANTIZED_FEW_ROWS: usize = 32;

/// Whether a packed product of `rows` rows goes to the few-rows race: up to
/// [`QUANTIZED_FEW_ROWS`], no rows included, the packed candidates' own admission answering every
/// width below the bound.
pub fn quantized_few_rows(rows: usize) -> bool {
    rows <= QUANTIZED_FEW_ROWS
}

impl MatmulAutotune {
    /// Which race a float product goes to, by [`float_few_rows`], so no width reaches a race
    /// whose every candidate declines it. Packed inputs decoded first make a float product here.
    pub fn new(problem: &MatmulProblem) -> Self {
        match problem.streamed_rows.is_some_and(float_few_rows) {
            true => Self::FewRows,
            false => Self::ManyRows,
        }
    }

    /// Which race a product reading packed inputs as stored goes to, by the rows alone
    /// ([`quantized_few_rows`]): the few-rows race's candidates decline what they do not close,
    /// and its floor, the packed inputs decoded, answers there.
    pub fn packed(problem: &MatmulProblem) -> Self {
        match problem.streamed_rows.is_some_and(quantized_few_rows) {
            true => Self::FewRows,
            false => Self::ManyRows,
        }
    }

    /// What [`PlanTree`](cubek_std::plan::PlanTree) calls this race, and the name a caller's
    /// tuner for it goes by.
    pub fn name(self) -> &'static str {
        match self {
            Self::FewRows => "few-rows",
            Self::ManyRows => "many-rows",
            Self::BlockScaled => "block-scaled",
        }
    }
}
