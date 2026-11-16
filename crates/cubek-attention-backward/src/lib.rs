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
//! A query's gradient sums over every key and a key's over every query. The strategies take that
//! two ways, each launching its routines in order, one kernel each.
//!
//! The unit, plane and cube staged strategies walk the tiles twice: once owned by the queries, for
//! their gradient and their `delta`, then owned by the keys, for theirs and their values'. Nothing
//! is summed across owners, so a run's gradients are the same bits every time.
//!
//! The hypercube strategy walks them once, owned by the keys. A first routine writes each query's
//! `delta`; then every cube of the walk adds its share of each query's gradient into one `f32`
//! sum, with atomic adds, and a last routine casts the sum into the query's gradient. A tile takes
//! five multiplications instead of seven, but the adds land in no set order, so the query's
//! gradient may differ from one run to the next in its last bits; the key's and the value's do
//! not. A device without `f32` atomic adds declines it.
//!
//! The routines, one folder per scope, and the strategies are in [`kernels`];
//! [`attention_backward`] runs the strategy and setting autotune measures fastest, the hypercube
//! strategy among them: no setting leaves it out.

#[cfg(feature = "benchmarks")]
pub mod eval;
pub mod kernels;

pub use kernels::{
    ATTENTION_BACKWARD_ELEMENT_UNVERIFIED, ATTENTION_BACKWARD_GROUP_UNEVEN,
    AttentionBackwardGradients, AttentionBackwardTensors, attention_backward,
};
