use cubecl;
use cubecl::prelude::*;

use crate::components::global::simple::AttentionWriter;
use crate::components::stage::Reducer;

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait AttentionPartitioner: Send + Sync + 'static {
    type Reducer: Reducer;
    type Writer<ES: Float, ESS: Size, EG: Float, EGS: Size>: AttentionWriter<ES, ESS, EG, EGS>;

    fn seq_q_index() -> u32;
}
