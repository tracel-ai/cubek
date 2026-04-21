//! Per-routine TilingScheme sweeps. For each routine family, a handful of
//! hand-picked (tile, partition, stage) combos covering:
//!   - 1x1x1 minimal partition/stage
//!   - k-reduction partition (k > 1)
//!   - multi-plane stage
//!
//! Only one representative backend per routine is covered here; backend-level
//! coverage lives in `normal/` and the full cartesian in `full/`.

use cubek_matmul::{
    launch::{Strategy, test_only::TestStrategy},
    routines::BlueprintStrategy,
};
use cubek_std::{PartitionSize, StageSize};

use super::common::{
    client, default_tile_size, f16_elems, plane_blueprint, problem, row_row,
};
use crate::suite::{test_matmul_strategy, test_matmul_test_strategy};

fn run_plane(strategy: impl FnOnce(cubek_matmul::definition::TilingBlueprint) -> Strategy, partition: PartitionSize, stage: StageSize) {
    let c = client();
    let p = problem(256, 256, 256, row_row(), f16_elems());
    let bp = plane_blueprint(&c, &p, default_tile_size(), partition, stage);
    test_matmul_strategy(c, p, strategy(bp));
}

fn run_plane_test_only(
    strategy: impl FnOnce(cubek_matmul::definition::TilingBlueprint) -> TestStrategy,
    partition: PartitionSize,
    stage: StageSize,
) {
    let c = client();
    let p = problem(256, 256, 256, row_row(), f16_elems());
    let bp = plane_blueprint(&c, &p, default_tile_size(), partition, stage);
    test_matmul_test_strategy(c, p, strategy(bp));
}

// -- Simple cyclic (Cmma representative) -------------------------------------

#[test]
fn simple_cyclic_cmma_partition_1x1x1_stage_1x1x1() {
    run_plane(
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 1, n: 1, k: 1 },
    );
}

#[test]
fn simple_cyclic_cmma_partition_1x1x4_stage_1x1x1() {
    run_plane(
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 4 },
        StageSize { m: 1, n: 1, k: 1 },
    );
}

#[test]
fn simple_cyclic_cmma_partition_2x1x4_stage_2x2x1() {
    run_plane(
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 2, n: 1, k: 4 },
        StageSize { m: 2, n: 2, k: 1 },
    );
}

// -- Double cyclic (Cmma) -----------------------------------------------------

#[test]
fn double_cyclic_cmma_stage_2x2x1() {
    run_plane(
        |bp| Strategy::DoubleCyclicCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
    );
}

#[test]
fn double_cyclic_cmma_partition_1x1x4_stage_1x1x1() {
    run_plane(
        |bp| Strategy::DoubleCyclicCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 4 },
        StageSize { m: 1, n: 1, k: 1 },
    );
}

// -- Ordered double (Cmma) ---------------------------------------------------

#[test]
fn ordered_double_cmma_stage_4x4x1() {
    run_plane(
        |bp| Strategy::OrderedDoubleCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 4, n: 4, k: 1 },
    );
}

// -- Specialized (Cmma) ------------------------------------------------------

#[test]
fn specialized_cyclic_cmma_stage_2x2x1() {
    run_plane(
        |bp| Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
    );
}

// -- Simple unit -------------------------------------------------------------

#[test]
fn simple_unit_partition_1x1x1() {
    let c = client();
    let p = problem(64, 64, 64, row_row(), f16_elems());
    let bp = plane_blueprint(
        &c,
        &p,
        cubek_std::TileSize { m: 4, n: 4, k: 4 },
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 1, n: 1, k: 1 },
    );
    test_matmul_strategy(c, p, Strategy::SimpleUnit(BlueprintStrategy::Forced(bp)));
}

#[test]
fn simple_unit_partition_2x2x1() {
    let c = client();
    let p = problem(64, 64, 64, row_row(), f16_elems());
    let bp = plane_blueprint(
        &c,
        &p,
        cubek_std::TileSize { m: 4, n: 4, k: 4 },
        PartitionSize { m: 2, n: 2, k: 1 },
        StageSize { m: 1, n: 1, k: 1 },
    );
    test_matmul_strategy(c, p, Strategy::SimpleUnit(BlueprintStrategy::Forced(bp)));
}

// -- Double unit -------------------------------------------------------------

#[test]
fn double_unit_partition_1x1x1_stage_2x2x1() {
    let c = client();
    let p = problem(64, 64, 64, row_row(), f16_elems());
    let bp = plane_blueprint(
        &c,
        &p,
        cubek_std::TileSize { m: 4, n: 4, k: 4 },
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
    );
    test_matmul_strategy(c, p, Strategy::DoubleUnit(BlueprintStrategy::Forced(bp)));
}

// -- Interleaved (test-only) -------------------------------------------------

#[test]
fn interleaved_partition_1x1x1() {
    let c = client();
    let p = problem(64, 64, 64, row_row(), f16_elems());
    let bp = plane_blueprint(
        &c,
        &p,
        cubek_std::TileSize { m: 4, n: 4, k: 4 },
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 1, n: 1, k: 1 },
    );
    test_matmul_test_strategy(c, p, TestStrategy::Interleaved(BlueprintStrategy::Forced(bp)));
}

// -- Simple barrier cooperative (test-only) ----------------------------------

#[test]
fn simple_barrier_cooperative_cmma_partition_1x1x1() {
    run_plane_test_only(
        |bp| TestStrategy::SimpleBarrierCooperativeCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 1, n: 1, k: 1 },
    );
}

// -- Simple barrier cyclic (test-only) ---------------------------------------

#[test]
fn simple_barrier_cyclic_cmma_stage_2x2x1() {
    run_plane_test_only(
        |bp| TestStrategy::SimpleBarrierCyclicCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
    );
}

// -- TMA ---------------------------------------------------------------------

#[test]
fn simple_tma_cmma_stage_2x2x1() {
    run_plane(
        |bp| Strategy::SimpleTmaCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
    );
}

#[test]
fn double_tma_cmma_stage_2x2x1() {
    run_plane(
        |bp| Strategy::DoubleTmaCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
    );
}

// -- Plane vecmat (VecMat) ---------------------------------------------------

#[test]
fn simple_vecmat_partition_1x1x2() {
    let c = client();
    let p = problem(1, 256, 256, row_row(), f16_elems());
    let bp = plane_blueprint(
        &c,
        &p,
        cubek_std::TileSize { m: 1, n: 4, k: 128 },
        PartitionSize { m: 1, n: 1, k: 2 },
        StageSize { m: 1, n: 1, k: 1 },
    );
    test_matmul_strategy(c, p, Strategy::SimpleVecMat(BlueprintStrategy::Forced(bp)));
}
