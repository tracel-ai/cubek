//! Forced-blueprint tests for non-default hypercube / swizzle / specialization
//! / partition-buffering knobs. All of these are applied to one representative
//! routine (SimpleCyclicCmma or DoubleCyclicCmma as appropriate) — the point
//! is to exercise each knob at least once, not to cover every combo.

use cubek_matmul::{
    components::{
        global::{InputLoadFlow, LoadFlows},
        stage::PartitionBuffering,
    },
    definition::SwizzleModes,
    launch::Strategy,
    routines::BlueprintStrategy,
};
use cubek_std::{
    PartitionSize, StageSize,
    cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint, SmAllocation},
    stage::SwizzleMode,
};

use super::common::{
    client, default_hypercube, default_tile_size, f16_elems, plane_blueprint_with, problem, row_row,
};
use crate::suite::test_matmul_strategy;

fn run_with(
    swizzle: SwizzleModes,
    hypercube: HypercubeBlueprint,
    buffering: PartitionBuffering,
    specialization: LoadFlows,
    strategy: impl FnOnce(cubek_matmul::definition::TilingBlueprint) -> Strategy,
) {
    let c = client();
    let p = problem(256, 256, 256, row_row(), f16_elems());
    let bp = plane_blueprint_with(
        &c,
        &p,
        default_tile_size(),
        PartitionSize { m: 1, n: 1, k: 1 },
        StageSize { m: 2, n: 2, k: 1 },
        swizzle,
        hypercube,
        buffering,
        specialization,
    );
    test_matmul_strategy(c, p, strategy(bp));
}

fn default_swizzle() -> SwizzleModes {
    SwizzleModes {
        lhs: SwizzleMode::None,
        rhs: SwizzleMode::None,
        ..Default::default()
    }
}

fn both_main() -> LoadFlows {
    LoadFlows {
        lhs: InputLoadFlow::MainOnly,
        rhs: InputLoadFlow::MainOnly,
    }
}

// -- Swizzle modes -----------------------------------------------------------

#[test]
fn swizzle_b32() {
    run_with(
        SwizzleModes {
            lhs: SwizzleMode::B32,
            rhs: SwizzleMode::B32,
            ..Default::default()
        },
        default_hypercube(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn swizzle_b64() {
    run_with(
        SwizzleModes {
            lhs: SwizzleMode::B64,
            rhs: SwizzleMode::B64,
            ..Default::default()
        },
        default_hypercube(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn swizzle_b128() {
    run_with(
        SwizzleModes {
            lhs: SwizzleMode::B128,
            rhs: SwizzleMode::B128,
            ..Default::default()
        },
        default_hypercube(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

// -- Hypercube global order --------------------------------------------------

#[test]
fn hypercube_swizzle_col() {
    run_with(
        default_swizzle(),
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleCol(2))
            .cube_count_strategy(CubeCountStrategy::FromProblem)
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn hypercube_col_flattened() {
    run_with(
        default_swizzle(),
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::ColMajor)
            .cube_count_strategy(CubeCountStrategy::Flattened)
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn hypercube_sm_exact() {
    run_with(
        default_swizzle(),
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::RowMajor)
            .cube_count_strategy(CubeCountStrategy::Sm {
                num_sms: 4,
                sm_usage: SmAllocation::Exact,
                cubes_first: false,
            })
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn hypercube_spread() {
    run_with(
        default_swizzle(),
        HypercubeBlueprint::builder()
            .global_order(GlobalOrder::SwizzleRow(2))
            .cube_count_strategy(CubeCountStrategy::Spread)
            .build(),
        PartitionBuffering::Single,
        both_main(),
        |bp| Strategy::SimpleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

// -- Load specialization (applied on a routine that supports it) -------------

#[test]
fn specialization_main_load() {
    run_with(
        default_swizzle(),
        default_hypercube(),
        PartitionBuffering::Single,
        LoadFlows {
            lhs: InputLoadFlow::MainOnly,
            rhs: InputLoadFlow::LoadOnly,
        },
        |bp| Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn specialization_load_main() {
    run_with(
        default_swizzle(),
        default_hypercube(),
        PartitionBuffering::Single,
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::MainOnly,
        },
        |bp| Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

#[test]
fn specialization_load_load() {
    run_with(
        default_swizzle(),
        default_hypercube(),
        PartitionBuffering::Single,
        LoadFlows {
            lhs: InputLoadFlow::LoadOnly,
            rhs: InputLoadFlow::LoadOnly,
        },
        |bp| Strategy::SpecializedCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}

// -- Partition buffering -----------------------------------------------------

#[test]
fn partition_buffering_double() {
    run_with(
        default_swizzle(),
        default_hypercube(),
        PartitionBuffering::Double,
        both_main(),
        |bp| Strategy::DoubleCyclicCmma(BlueprintStrategy::Forced(bp)),
    );
}
