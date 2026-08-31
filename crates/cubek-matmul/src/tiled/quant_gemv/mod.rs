//! The quantized decode gemv: `y = W · x` with `W` packed and its scales an operand of their own.
//!
//! The weight's physical `[d_out, d_in]` buffer is the **lhs**, so the contraction runs along the
//! buffer's contiguous direction — the orientation a decode step streams. `K` is spelled as the
//! two axes a scale block makes of it, `(KB, KI)`: one scale per block is then the scales operand
//! omitting `KI`, and nothing anywhere divides a position by a block size.

mod base;
mod kernel;
mod launch;
mod operands;

pub use base::{QuantGemvBlueprint, QuantGemvProblem, QuantGemvRoutine, QuantGemvStrategy};
pub use launch::{QuantGemvBindings, QuantGemvElems, launch_ref};
