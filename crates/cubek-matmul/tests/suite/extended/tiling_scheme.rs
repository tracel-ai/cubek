//! Per-routine TilingScheme sweeps. For each routine family, a handful of
//! hand-picked (tile, partition, stage) combos covering:
//!   - 1x1x1 minimal partition/stage
//!   - k-reduction partition (k > 1)
//!   - multi-plane stage
//!
//! Only one representative backend per routine is covered here; backend-level
//! coverage lives in `normal/` and the full cartesian in `full/`.

use cubek_matmul::{
    definition::MatmulProblem,
    launch::{Strategy, test_only::TestStrategy},
    routines::BlueprintStrategy,
};
use cubek_std::{MatrixLayout, PartitionSize, StageSize, TileSize};

use super::common::{client, default_tile_size, f16_elems, plane_blueprint, problem, row_row};
use crate::suite::{extended::test_matmul_test_strategy, test_matmul_strategy};

fn run_plane(
    strategy: impl FnOnce(cubek_matmul::definition::TilingBlueprint) -> Strategy,
    partition: PartitionSize,
    stage: StageSize,
) {
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

// Unit-partitioned routines need the number of units inside a stage to be a
// multiple of plane_dim (32 on most wgpu targets). With the register tile
// matmul that means stage.m * stage.n must be divisible by 32, so we size the
// stage accordingly.
fn unit_stage() -> StageSize {
    StageSize { m: 8, n: 4, k: 1 }
}

fn unit_tile() -> TileSize {
    TileSize { m: 4, n: 4, k: 4 }
}

fn unit_problem() -> MatmulProblem {
    problem(64, 64, 64, row_row(), f16_elems())
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
//
// Ordered requires `partitions_per_stage_along_n == 1`, so stage.n is fixed
// at 1 here.

#[test]
fn ordered_double_cmma_stage_4x1x1() {
    run_plane(
        |bp| Strategy::OrderedDoubleCmma(BlueprintStrategy::Forced(bp)),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 4, n: 1, k: 1 },
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
    let p = unit_problem();
    let bp = plane_blueprint(
        &c,
        &p,
        unit_tile(),
        PartitionSize { m: 1, n: 1, k: 1 },
        unit_stage(),
    );
    test_matmul_strategy(c, p, Strategy::SimpleUnit(BlueprintStrategy::Forced(bp)));
}

#[test]
fn simple_unit_partition_2x2x1() {
    let c = client();
    let p = unit_problem();
    let bp = plane_blueprint(
        &c,
        &p,
        unit_tile(),
        PartitionSize { m: 2, n: 2, k: 1 },
        unit_stage(),
    );
    test_matmul_strategy(c, p, Strategy::SimpleUnit(BlueprintStrategy::Forced(bp)));
}

// -- Double unit -------------------------------------------------------------

#[test]
fn double_unit_partition_1x2x1() {
    let c = client();
    let p = unit_problem();
    let bp = plane_blueprint(
        &c,
        &p,
        unit_tile(),
        // partition.n=2 is required for partition-level double buffering.
        PartitionSize { m: 1, n: 2, k: 1 },
        unit_stage(),
    );
    test_matmul_strategy(c, p, Strategy::DoubleUnit(BlueprintStrategy::Forced(bp)));
}

// -- Interleaved (test-only) -------------------------------------------------
//
// Interleaved tile matmul is picky: `tile.k` must be a multiple of plane_dim
// (typically 32), and the k-local chunk (`tile.k / plane_dim`) must in turn be
// a multiple of the lhs vector size (typically 4). That pushes tile.k to 128
// on a plane_dim=32 runtime.

#[test]
fn interleaved_partition_1x1x1() {
    let c = client();
    let p = problem(64, 64, 128, row_row(), f16_elems());
    let bp = plane_blueprint(
        &c,
        &p,
        TileSize { m: 4, n: 4, k: 128 },
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 1, n: 1, k: 1 },
    );
    test_matmul_test_strategy(
        c,
        p,
        TestStrategy::Interleaved(BlueprintStrategy::Forced(bp)),
    );
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
//
// The plane vecmat tile matmul only supports a ColMajor rhs, so this test is
// the one place in the extended tier that uses a non-row-row layout.

#[test]
fn simple_vecmat_partition_1x1x2() {
    let c = client();
    let p = problem(
        1,
        256,
        256,
        (MatrixLayout::RowMajor, MatrixLayout::ColMajor),
        f16_elems(),
    );
    let bp = plane_blueprint(
        &c,
        &p,
        TileSize { m: 1, n: 4, k: 128 },
        PartitionSize { m: 1, n: 1, k: 2 },
        StageSize { m: 1, n: 1, k: 1 },
    );
    test_matmul_strategy(c, p, Strategy::SimpleVecMat(BlueprintStrategy::Forced(bp)));
}
