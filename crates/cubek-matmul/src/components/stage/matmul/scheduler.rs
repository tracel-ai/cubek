//! Re-export shim. The partition scheduler now lives in
//! `cubek_std::tile::variants::stage::scheduler`; this file keeps the
//! existing `cubek_matmul::components::stage::matmul::scheduler` path
//! working for callers. Deleted in PR 6 once consumers update their
//! imports.

pub use cubek_std::tile::{PartitionScheduler, PartitionSchedulerScheme};
