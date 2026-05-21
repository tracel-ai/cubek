//! Per-variant tile payloads, configs, and per-variant compute. Cross-variant
//! dispatch lives in [`crate::tile::ops`].

pub(crate) mod bounce;
pub(crate) mod instruction;
pub(crate) mod row_wise;
pub(crate) mod shared;
pub(crate) mod stage;
pub(crate) mod unit;
pub(crate) mod whitebox_fragment;

pub use bounce::*;
pub use instruction::*;
pub use row_wise::*;
pub use shared::*;
pub use stage::*;
pub use unit::*;
pub use whitebox_fragment::*;
