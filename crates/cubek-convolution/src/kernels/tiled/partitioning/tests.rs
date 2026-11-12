use super::ConvTiles;
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::Conv2dProblem;
use crate::kernels::tiled::routines::CubeStagedConvRoutine;
use crate::kernels::tiled::routines::test_support::{dense, nvidia};
use crate::kernels::tiled::{ConvProblem, ConvSettings};
use crate::launch::ConvolutionArgs;
use cubek_std::routine::Routine;

const BUDGET: usize = 48 * 1024;

/// The tiles of the Mma 2x2 grid on `problem`.
fn tiles(problem: &ConvProblem) -> ConvTiles {
    let settings = CubeStagedConvRoutine::F2X2_P2X2_T16.mma();
    let plane = settings
        .plane(&nvidia(), problem)
        .expect("the device offers mma");
    ConvTiles::new(problem, &plane, &settings).expect("the tiles state")
}

/// The steps a stage of the Mma 2x2 grid holds on `problem`, within `budget`.
fn steps(problem: &ConvProblem, budget: usize) -> usize {
    let tiles = tiles(problem);
    let storage = tiles.storage(problem);
    tiles.stage(&nvidia(), problem, &storage, budget).0.1
}

/// The forward of three channels under an 11 wide filter at stride four, six steps of eight.
fn alexnet_like() -> ConvProblem {
    ConvProblem::from(&Conv2dProblem {
        input_shape: [8, 3, 227, 227],
        weight_shape: [64, 3, 11, 11],
        args: ConvolutionArgs {
            stride: [4; 2],
            ..dense().args
        },
        ..dense()
    })
}

/// A weight backward whose walk is seven steps of eight, a prime count, takes it whole as one
/// stage.
#[test]
fn a_prime_walk_is_one_stage() {
    let problem = ConvProblem::from(&Conv2dProblem {
        operation: ConvolutionOperation::BackwardWeight,
        ..dense()
    });
    assert_eq!(steps(&problem, BUDGET), 7);
    assert_eq!(steps(&problem, 1024), 1);
}

/// A second stage in flight doubles the shared memory the stages hold.
#[test]
fn a_double_ring_holds_twice_the_bytes() {
    let problem = ConvProblem::from(&dense());
    let bytes = |settings: ConvSettings| {
        let (_, grid) = CubeStagedConvRoutine
            .prepare(&nvidia(), &problem, &settings)
            .expect("the setting prepares");
        grid.shared_memory
    };
    let single = CubeStagedConvRoutine::F2X2_P2X2_T16.mma();
    assert_eq!(bytes(single.double_ring()), 2 * bytes(single));
}

/// A contraction that fits one stage has no second to fill ahead, so a double ring holds one.
#[test]
fn a_double_ring_over_one_stage_holds_one() {
    let problem = ConvProblem::from(&Conv2dProblem {
        input_shape: [1, 16, 8, 8],
        weight_shape: [16, 16, 1, 1],
        ..dense()
    });
    let ring = |settings: ConvSettings| {
        let (blueprint, _) = CubeStagedConvRoutine
            .prepare(&nvidia(), &problem, &settings)
            .expect("the setting prepares");
        blueprint.microkernel.ring
    };
    let single = CubeStagedConvRoutine::F2X2_P2X2_T16.mma();
    assert_eq!(ring(single.double_ring()), ring(single));
}

/// A walk with a divisor in reach keeps it: six steps of the AlexNet-like forward run three
/// at a time.
#[test]
fn a_walk_with_a_divisor_keeps_it() {
    let problem = alexnet_like();
    assert_eq!(steps(&problem, BUDGET), 3);
}

/// A budget that holds two steps but not three takes a walk of six down from three to two.
#[test]
fn a_stage_walks_down_to_the_budget() {
    let problem = alexnet_like();
    let tiles = tiles(&problem);
    let storage = tiles.storage(&problem);
    let two = tiles.stage_bytes(&nvidia(), problem.elements, &storage, 2);
    assert_eq!(steps(&problem, two), 2);
}
