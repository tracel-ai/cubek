//! What kernels are built from.
//!
//! A **microkernel** is what one owner of a work item runs, and its **scope** is the owner: a
//! unit, a plane, a cube. Owners at one scope never share a work item; the units inside one owner
//! all work on the same one. A microkernel may be built from microkernels of a smaller scope: a
//! cube's planes each running a plane microkernel on part of the cube's item.
//!
//! Nothing here is launched. A [`Routine`](crate::routine::Routine) is a whole kernel written
//! around one microkernel, and the launch is its own.
//!
//! What every family's staged walk shares, whatever operands it stages, is here too: the
//! [`StageRing`] that holds its stages, and what a stage of each operand costs.
//!
//! Autotune prioritizes routines, never microkernels. A routine and the microkernels under it each
//! estimate their own part of the kernel's performance into a [`KernelEstimate`], in quantities
//! every routine states alike.

mod base;
mod cube_planes;
mod estimate;
mod staging;

pub use base::*;
pub use cube_planes::*;
pub use estimate::*;
pub use staging::*;
