//! How an operand's logical [`Space`](crate::Space) axes map onto its buffer's
//! [`PhysicalAxis`](crate::PhysicalAxis) positions.
//!
//! The default is one logical axis per physical axis with coefficient `1`: logical rank equals
//! physical rank, one tile coordinate moves one physical axis by one sub-tile, and sibling windows
//! never overlap. Matmul fits because `K` genuinely is a physical axis of `lhs` and `rhs`.
//!
//! A gather-reduce over an *abstract* dimension does not fit. A stencil's reduce axes address the
//! same input physical axis its output axes address, each with its own coefficient:
//!
//! ```text
//! space {Oh, Ow, Cout, Rh, Rw, Cin}
//!   input physical axis Ih <- Oh*stride_h + Rh*dilation_h
//!         physical axis Iw <- Ow*stride_w + Rw*dilation_w
//!         physical axis Cin
//! ```
//!
//! So a physical axis is an affine combination of logical axes, and consecutive windows along `Oh`
//! overlap by the receptive field. [`Projection`] is that combination;
//! [`direct`](Projection::direct) is the degenerate one every current operand uses.
//!
//! A projection is written one buffer dim at a time ([`Projection::dims`], in [`build`]) and,
//! once the buffer's real extents and strides are in hand, asked which axis is contiguous, how
//! far an axis steps, and whether the dims alias or run out of order ([`query`]). It holds no
//! extents and no strides of its own: which dim carries `k` is a fact about the operand, and
//! how far `k` steps is a fact about the allocation, and they arrive from different places.

mod base;
mod build;
mod carrier;
mod compact;
mod fold;
mod map;
mod query;
mod tiling;

pub use base::*;
pub use build::*;
pub use compact::*;
pub use fold::*;
pub use map::*;
pub use query::*;
pub use tiling::*;

/// Shared by the two places a set of coefficients has a common factor worth taking out:
/// [`PhysicalAxisMap::over`], where a divisor every coefficient cancels is not a division at all,
/// and [`Compaction`], where the step it leaves is what a stage stores instead of the whole box.
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}
