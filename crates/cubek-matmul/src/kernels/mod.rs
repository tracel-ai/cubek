//! Matmul kernels built on the microkernels in [`microkernel`](cubek_microkernel::matmul): the dense
//! product's routines ([`MatmulRoutine`]) and which kernel runs a product ([`select`]). A routine
//! reads each operand as it is stored ([`tensor`]).
//!
//! There is one product. A product whose inputs read a table, one entry a batch ([`MatmulIndex`]),
//! is the dense product over those batches.
//!
//! cubek measures nothing. A caller that does races the settings of the routines it lists, float
//! ([`MatmulRoutineCandidate`]) and packed ([`MatmulPackedCandidate`]), on a call keyed by
//! [`MatmulTuneKey`], names a winner by its [`MatmulRoutineSettings`], and hands [`select`] the
//! plan of the race a product goes to ([`MatmulAutotune`](select::MatmulAutotune)).

pub mod plan;
pub mod select;
pub mod tensor;

mod candidate;
mod dense;
mod key;
mod listing;
mod packed_candidate;
mod routine_prepared;
mod routine_settings;

pub use candidate::*;
pub use dense::*;
pub use key::*;
pub use listing::{
    FLOAT_SETTING_PACKED_INPUT, ListedFor, Listing, PACKED_SETTING_FLOAT_INPUTS, Product,
};
pub use packed_candidate::*;
pub use routine_prepared::*;
pub use routine_settings::*;
pub use select::{MatmulKernel, MatmulKernelHeuristic, MatmulOpStrategy};
