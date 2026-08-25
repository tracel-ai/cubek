//! Broadcast (stride-0) coverage for the tile-DSL strategies.

use cubek_matmul::{routine::BlueprintStrategy, tiled::Strategy as Tiled};

use crate::harness::assert_batch_broadcast;

#[test]
fn batch_broadcast_cpu_gemm() {
    assert_batch_broadcast(Tiled::CpuGemm(BlueprintStrategy::default()).into());
}
