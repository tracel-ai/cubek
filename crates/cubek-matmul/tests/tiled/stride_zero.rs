//! Broadcast (stride-0) coverage for the tile-DSL strategies.

use cubek_matmul::{kernels::UnitRegistersMatmulRoutineSettings, tiled::Strategy as Tiled};

use crate::harness::assert_batch_broadcast;

#[test]
fn batch_broadcast_cpu_gemm() {
    assert_batch_broadcast(Tiled::CpuGemm(UnitRegistersMatmulRoutineSettings::DEFAULT).into());
}
