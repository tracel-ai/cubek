//! The K-grouped product: two operands' rows in contiguous groups, each group's rows of one
//! contracted against the same rows of the other into its own slab of a stack, launched as the
//! dense product of its groups, each a batch summing its run of rows. The M-grouped product's
//! gradient for its stack.

mod launch;
mod tensors;

#[cfg(test)]
mod tests;

pub use launch::*;
pub use tensors::*;
