//! The kernel arguments that carry operands, host and device side.

pub(crate) mod accumulate;
pub(crate) mod analysis;
pub(crate) mod base;
pub(crate) mod delivery;
pub(crate) mod destination;
pub(crate) mod operand;
pub(crate) mod relay;
pub(crate) mod scale;
pub(crate) mod tensor;
pub(crate) mod tma;

pub(crate) use base::*;
