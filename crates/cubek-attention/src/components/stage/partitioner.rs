use cubecl;
use cubecl::prelude::*;

use crate::components::global::simple::AttentionWriter;

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and compute indexing.
pub trait AttentionPartitioner: Send + Sync + 'static {
    type Writer<ES: Float, EG: Float>: AttentionWriter<ES, EG>;

    fn seq_q_index() -> u32;
}
