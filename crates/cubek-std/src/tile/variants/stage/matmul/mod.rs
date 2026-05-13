//! Partition-matmul body and per-partition fragment helpers. Migrated from
//! `cubek_matmul::components::stage::matmul::partition`.

pub(crate) mod body;
pub(crate) mod fragments;

pub use body::*;
pub use fragments::*;
