use cubecl as cubecl;
use cubecl::prelude::*;
use cubecl::std::tensor::layout::Coords1d;

use crate::components::global::simple::AttentionWriter;
use crate::components::stage::Reducer;

#[cube]
/// Defines how the stage is partitioned among compute primitives (e.g., units or planes).
/// Controls global writeback and and compute indexing.
pub trait AttentionPartitioner: Send + Sync + 'static {
    type Reducer: Reducer;
    type Writer<ES: Float, EG: Float>: AttentionWriter<ES, EG>;

    fn seq_q_index() -> Coords1d;
}
