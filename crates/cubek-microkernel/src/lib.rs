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
//! A folder per problem, behind its family's feature: what the problem is in its `base`, beside a
//! folder per scope. What every family shares, as the ring a staged walk holds its stages in, is
//! cubek-std's.

#[cfg(feature = "attention")]
pub mod attention;
#[cfg(feature = "attention-backward")]
pub mod attention_backward;
#[cfg(feature = "matmul")]
pub mod matmul;
#[cfg(feature = "attention")]
pub mod split_keys_merge;
