//! The software contraction nest `acc += lhs · rhs`, in its two accumulator forms. The peer of
//! the hardware leaves in `instruction/mma`, reached from the same dispatch.
//!
//! `base` is the memory form's entry point (`memory`), for a plain `Gmem`/`Smem` accumulator:
//! it resolves each operand's quant packing, then routes to `direct` (the 2-D nest, a single
//! contracted axis off directly addressed operands) or `gather` (the N-D nest, multiple
//! contracted axes or gathered operands). `promoted` is the register-resident form, whose
//! block outlives the call.

mod base;
mod direct;
mod gather;
mod promoted;
mod shape;

pub use base::ScaleSide;
pub(crate) use base::{memory, memory_scaled, scale_side};
