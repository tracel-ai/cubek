//! What one plane contracts through, at a tile no stated level cuts further: the two hardware
//! fragments ([`cmma`], [`mma`]), and the software register nest ([`registers`]) in its two
//! accumulator forms, one holding its block across calls ([`promoted`]) and one seeding it from
//! memory per visit ([`memory`]).

mod cmma;
pub(crate) mod memory;
mod mma;
mod promoted;
mod registers;
mod scale;

pub(crate) use cmma::rhs_layout;
pub use scale::Side;
pub(crate) use scale::{check_scales_omit_rather_than_divide, check_scales_ride};
