//! The attention microkernels: the queries over the keys and values,
//! `softmax(query · keyᵀ · scale) · value`, a folder per scope ([`unit`](mod@unit), [`plane`],
//! [`cube`], [`hypercube`]). Beside them, what the problem is, which all of them and their kernels
//! read: the [`AttentionWorkItem`], its axes and operands, which keys a query sees, and the
//! [`AttentionMicrokernel`] trait, each handed the space its cube covers; where the cubes lie is
//! the kernels' placement. Where workers split a row's keys, their partials are merged by
//! [`split_keys_merge`](crate::split_keys_merge).

mod base;
pub mod cube;
pub mod hypercube;
pub mod plane;
pub mod unit;

pub use base::*;
