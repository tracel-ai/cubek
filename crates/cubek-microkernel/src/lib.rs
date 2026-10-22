//! The microkernels every kernel family is built from.
//!
//! A **microkernel** is what one owner of a work item runs, and its **scope** is the owner: a
//! unit, a plane, a cube. Owners at one scope never share a work item; the units inside one owner
//! all work on the same one. A microkernel may be built from microkernels of a smaller scope — a
//! cube's planes each running a plane microkernel on part of the cube's item — or of another
//! problem: an attention scores its keys with a matmul microkernel.
//!
//! Nothing here is launched. A kernel places microkernels over a grid, binds its tensors and
//! launches; the kernels are their family's crate's (`cubek-matmul`'s `kernels`). The framework
//! every microkernel shares, the [`Microkernel`](cubek_std::microkernel::Microkernel) trait and
//! its estimate, is cubek-std's.
//!
//! A folder per problem, behind a feature of the same name, and the stage ring every staged walk
//! holds its stages in ([`staging`]).

#[cfg(feature = "matmul")]
pub mod matmul;
pub mod staging;
