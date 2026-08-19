//! The software contraction nest: `acc += lhs · rhs` into a plain `Gmem`/`Smem` accumulator, the
//! peer of the hardware leaves in [`instruction::mma`](crate::instruction::mma).
//!
//! `base` is the entry point (`memory`): it resolves each operand's quant packing, then
//! routes to `direct` (the 2-D nest, a single contracted axis off directly addressed operands) or
//! `gather` (the N-D nest, multiple contracted axes or gathered operands).

mod base;
mod direct;
mod gather;

pub(crate) use base::memory;
