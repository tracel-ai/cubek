//! The microkernels every kernel family is built from. What a microkernel and its scope are, and
//! the [`Microkernel`](cubek_std::microkernel::Microkernel) trait and estimate every one shares,
//! are cubek-std's ([`microkernel`](cubek_std::microkernel)). Beyond a smaller scope's, a
//! microkernel here may be built from microkernels of another problem: an attention scores its
//! keys with a matmul microkernel.
//!
//! Nothing here is launched. A kernel places microkernels over a grid, binds its tensors and
//! launches; the kernels are their family's crate's (`cubek-matmul`'s and `cubek-attention`'s
//! `kernels`).
//!
//! A folder per problem, behind a feature of the same name, and the stage ring every staged walk
//! holds its stages in ([`staging`]).

#[cfg(feature = "attention")]
pub mod attention;
#[cfg(feature = "attention-backward")]
pub mod attention_backward;
#[cfg(feature = "matmul")]
pub mod matmul;
pub mod staging;
