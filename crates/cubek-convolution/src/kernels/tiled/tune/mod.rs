//! How the tiled convolution is measured: the key that makes two convolutions the same problem,
//! the candidate each listed setting is raced as, and the autotune a launch without a setting runs.

mod base;
mod candidate;
mod key;

pub use base::*;
pub use candidate::*;
pub use key::*;

#[cfg(test)]
mod tests;
