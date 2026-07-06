//! The opaque axis label every layout/space concept is keyed on. A client gives it
//! meaning (matmul's `M`/`N`/`K`, reduce's reduce axis, …); the vocabulary stays agnostic.

/// Bump if a problem declares more.
pub const MAX_AXES: usize = 6;

/// A labeled axis. The `u8` is a client-assigned index, not a position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Axis(pub u8);
