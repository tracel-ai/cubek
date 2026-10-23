//! Attention's backward pass: the gradients of `out = softmax(query · keyᵀ · scale + bias) · value`
//! against the query, the keys and the values, from the loss's gradient against the output.
//!
//! On every tile of queries by keys the pass reads the probabilities back from the logsumexp the
//! forward left, `p = exp(query · keyᵀ · scale + bias − logsumexp)`, and takes
//!
//! ```text
//! value_gradient += pᵀ · out_gradient
//! score_gradient  = p ⊙ (out_gradient · valueᵀ − delta),  delta = Σ_dv out_gradient ⊙ out
//! query_gradient += scale · score_gradient · key
//! key_gradient   += scale · score_gradientᵀ · query
//! ```
//!
//! A query's gradient sums over every key and a key's over every query, so each routine walks the
//! tiles twice, in two launches: once owned by the queries, for their gradient and their `delta`,
//! then owned by the keys, for theirs and their values'. Nothing is summed across owners, so a
//! run's gradients are the same bits every time.
//!
//! The routines, one folder per scope, are in `kernels`; [`attention_backward`] runs the one
//! autotune measures fastest.

#[cfg(feature = "benchmarks")]
pub mod eval;
pub mod kernels;
pub mod tune;

mod launch;
mod problem;
mod routine;
mod tensors;

pub use launch::*;
pub use problem::*;
pub use routine::*;
pub use tensors::*;
