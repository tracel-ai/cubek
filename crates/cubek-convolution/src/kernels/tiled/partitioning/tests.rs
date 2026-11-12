use super::ConvTiles;
use crate::components::ConvolutionOperation;
use crate::eval::benchmarks::Conv2dProblem;
use crate::eval::benchmarks::settings;
use crate::kernels::tiled::ConvProblem;
use crate::kernels::tiled::routines::test_support::{dense, few_channels_deep_filter, nvidia};

const BUDGET: usize = 48 * 1024;

/// The tiles of the Mma 2x2 grid on `problem`.
fn tiles(problem: &ConvProblem) -> ConvTiles {
    let settings = settings::F2X2_P2X2_T16.mma();
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
    ConvProblem::from(&few_channels_deep_filter())
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
