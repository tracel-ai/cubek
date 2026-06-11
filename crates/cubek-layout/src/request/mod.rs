//! A strategy's intrinsic request for the physical (storage) layout of its operands.
//! Blind to problem extents and to the incoming layout, so it is a static property of a
//! strategy. This module is the vocabulary plus its feasibility/preference predicate;
//! realizing a request against real extents + the delivered layout lives elsewhere.

mod base;
mod concrete;
mod facet;

#[cfg(test)]
mod tests;

pub use base::{Constraint, LayoutRequest, Strength};
pub use concrete::{ConcreteLayout, PhysicalAxis};
pub use facet::{AxisSet, Facet};
