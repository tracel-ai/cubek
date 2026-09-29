//! Plane-level contraction leaves: hardware fragments ([`cmma`], [`mma`]) and the register nest
//! ([`registers`]) with its [`promoted`] and [`memory`] accumulator forms.

mod cmma;
pub(crate) mod memory;
mod mma;
mod promoted;
mod registers;
mod scale;

pub(crate) use cmma::rhs_layout;
pub use scale::Side;
pub(crate) use scale::{check_scales_omit_rather_than_divide, check_scales_ride};
