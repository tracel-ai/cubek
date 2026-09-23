//! The N-D nest: an operand whose contracted axes are gathered rather than lined, its coordinates
//! resolved per tap ([`coords`]), walked whole ([`nd`]) or one factor at a time ([`separable`]).

mod base;
mod coords;
mod nd;
mod separable;

pub(crate) use base::*;
