//! The software contraction nest `acc += lhs · rhs`, in its two accumulator forms. The peer of
//! the hardware leaves in `instruction/mma`, reached from the same dispatch.
//!
//! `base` is the memory form's entry point (`memory`) for a plain `Gmem`/`Smem` accumulator: it
//! resolves each operand's quant packing, then routes to `direct` (2-D, one contracted axis off
//! direct operands) or `gather` (N-D). `promoted`, the register form, keeps its block across calls.

mod base;
mod direct;
mod gather;
mod promoted;
mod scale;
mod shape;

pub(crate) use base::{contracted_per_step, memory};
pub use scale::Side;
pub(crate) use scale::{check_scales_omit_rather_than_divide, check_scales_ride};
