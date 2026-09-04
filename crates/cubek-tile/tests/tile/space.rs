//! Unit tests for [`Space`]

use cubecl::prelude::*;
use cubek_tile::{Axis, ByAxis, CubeAxis, Distribution, Partitioner, Space, Tiling, cubes, lanes};

// Matmul-style axis labels reused across the cases below. `B0`/`B1` are two
// independent batch axes (a batch is just ordinary axes; broadcasting is omission).
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B0: Axis = Axis(3);
const B1: Axis = Axis(4);

// ---- Space ----------------------------------------------------------------

#[test]
fn new_builds_plain_axes() {
    let space = Space::new(&[(M, 4), (N, 8), (K, 2)]);
    assert_eq!(space.rank(), 3);
    assert_eq!(space.extent(M), 4);
    assert_eq!(space.extent(N), 8);
    assert_eq!(space.extent(K), 2);
}

#[test]
fn project_keeps_listed_axes_in_order() {
    let space = Space::new(&[(B0, 12), (M, 16), (K, 8)]);
    let lhs = space.project(&[B0, M, K]);
    assert_eq!(lhs.rank(), 3);
    assert_eq!(lhs.extent(B0), 12);
    assert_eq!(lhs.extent(M), 16);

    // An operand broadcasts a batch axis by simply leaving it out of the projection.
    let dropped = space.project(&[M, K]);
    assert_eq!(dropped.rank(), 2);
    assert!(!dropped.contains(B0));
}

#[test]
fn merge_assembles_two_batch_broadcast() {
    // lhs carries B0, rhs carries B1; each omits (broadcasts) the other's batch axis.
    // The merge rebuilds the full {B0, B1, M, N, K} operation space.
    let lhs = Space::new(&[(B0, 4), (M, 16), (K, 8)]);
    let rhs = Space::new(&[(B1, 3), (K, 8), (N, 4)]);
    let out = Space::new(&[(B0, 4), (B1, 3), (M, 16), (N, 4)]);

    let merged = Space::merge(&[&out, &lhs, &rhs]);
    assert_eq!(merged.rank(), 5);
    assert_eq!(merged.extent(B0), 4);
    assert_eq!(merged.extent(B1), 3);
    assert_eq!(merged.extent(K), 8);
    // First-appearance order: out's axes lead, K (only on the operands) comes last.
    assert_eq!(merged.axis_at(0), B0);
    assert_eq!(merged.axis_at(1), B1);
    assert_eq!(merged.axis_at(4), K);
}

#[test]
fn merge_shared_size_one_axis_broadcasts() {
    // A shared axis where one side is size 1 broadcasts to the other (numpy rule).
    let lhs = Space::new(&[(B0, 1), (M, 16)]);
    let rhs = Space::new(&[(B0, 4), (M, 16)]);
    assert_eq!(Space::merge(&[&lhs, &rhs]).extent(B0), 4);
}

#[test]
fn merge_omitted_axis_broadcasts_wholesale() {
    let lhs = Space::new(&[(B0, 12), (M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 8), (N, 4)]);

    let merged = Space::merge(&[&lhs, &rhs]);
    assert!(merged.contains(B0));
    assert_eq!(merged.extent(B0), 12);
}

#[test]
fn merge_plain_shared_axis_agrees() {
    let lhs = Space::new(&[(M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 8), (N, 4)]);
    let merged = Space::merge(&[&lhs, &rhs]);
    assert_eq!(merged.extent(K), 8);
    assert_eq!(merged.rank(), 3);
}

#[test]
#[should_panic(expected = "conflicting extents")]
fn merge_conflicting_extent_panics() {
    let lhs = Space::new(&[(M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 4), (N, 4)]); // K disagrees: 8 vs 4
    let _ = Space::merge(&[&lhs, &rhs]);
}

// ---- Space::divide (the tiling scheme) ------------------------------------

fn sequential(edges: &[(Axis, usize)]) -> Partitioner {
    let dists = edges
        .iter()
        .map(|&(a, _)| (a, Distribution::Sequential))
        .collect::<Vec<_>>();
    Partitioner::over(ByAxis::new(edges), ByAxis::new(&dists)).level()
}

#[test]
fn divide_cuts_each_axis_to_its_sub_tile_edge() {
    let partitioner = sequential(&[(M, 4), (N, 3), (K, 2)]);
    let space = Space::new(&[(M, 16), (N, 12), (K, 8)]).with_partitioner(partitioner);

    let tile = space.divide();
    assert_eq!(tile.extent(M), 4);
    assert_eq!(tile.extent(N), 3);
    assert_eq!(tile.extent(K), 2);
    assert_eq!(tile.rank(), 3);
}

#[test]
fn divide_chains_into_a_multi_level_scheme() {
    // The scheme is a tree of spaces
    let space = Space::new(&[(M, 64), (N, 64)]).with_partitioner(sequential(&[(M, 16), (N, 16)]));
    let level1 = space.divide();
    let level2 = level1
        .clone()
        .with_partitioner(sequential(&[(M, 4), (N, 4)]))
        .divide();

    assert_eq!(level1.extent(M), 16);
    assert_eq!(level2.extent(M), 4);
    assert_eq!(level2.extent(N), 4);
}

// ---- Space::overhangs ------------------------------------------------------

/// A cpu_gemm-shaped two-level scheme: a cube tile of `planes × leaf` leaves over `(m, n, k)`,
/// K cut to its full extent at the cube level (sequential contraction) then to `leaf_k`.
fn cpu_gemm_space(m: usize, n: usize, k: usize) -> Space {
    let (leaf_m, leaf_n, leaf_k) = (8, 8, 4);
    let (planes_m, planes_n) = (2, 4);
    Space::new(&[(M, m), (N, n), (K, k)])
        .with_partitioner(sequential(&[
            (M, planes_m * leaf_m),
            (N, planes_n * leaf_n),
            (K, k),
        ]))
        .with_partitioner(sequential(&[(M, leaf_m), (N, leaf_n), (K, leaf_k)]))
}

#[test]
fn overhangs_matches_cpu_gemm_checks() {
    // Every level divides: cube tiles 16×32, leaves 8×8×4.
    let space = cpu_gemm_space(64, 64, 16);
    assert!(!space.overhangs(M));
    assert!(!space.overhangs(N));
    assert!(!space.overhangs(K));

    // m = 40 is not a multiple of the cube tile (16): M overhangs (cpu_gemm's check_m).
    // Within a cube the plane split is exact, so the leaf level adds nothing.
    assert!(cpu_gemm_space(40, 64, 16).overhangs(M));

    // K's cube-level cut is its full extent (always divides); k = 18 fails only at the
    // leaf (leaf_k = 4): the deeper level alone drives the overhang (cpu_gemm's check_k).
    let space = cpu_gemm_space(64, 64, 18);
    assert!(space.overhangs(K));
    assert!(!space.overhangs(M));
}

#[test]
fn overhangs_when_a_deeper_edge_misdivides_its_parent() {
    // Top divides (32 % 16 == 0) but the second edge doesn't divide the first (16 % 3 != 0):
    // the parent edge, not the top extent, is what each level must divide.
    let space = Space::new(&[(M, 32)])
        .with_partitioner(sequential(&[(M, 16)]))
        .with_partitioner(sequential(&[(M, 3)]));
    assert!(space.overhangs(M));
}

#[test]
fn overhangs_final_space_never() {
    // No partitioner level: nothing to misdivide.
    assert!(!Space::new(&[(M, 7)]).overhangs(M));
}

#[test]
#[should_panic(expected = "concrete space")]
fn overhangs_dynamic_axis_panics() {
    let space = Space::new(&[(M, 64)])
        .with_partitioner(sequential(&[(M, 16)]))
        .all_dynamic();
    let _ = space.overhangs(M);
}

#[test]
fn with_partitioner_stacks_levels_and_divide_descends() {
    // Stacking partitioners builds the whole multi-level scheme up front
    let space = Space::new(&[(M, 64), (N, 64)])
        .with_partitioner(sequential(&[(M, 16), (N, 16)]))
        .with_partitioner(sequential(&[(M, 4), (N, 4)]));
    assert!(!space.is_final());

    let level1 = space.divide(); // head (16×16) consumed, 4×4 remains
    assert_eq!(level1.extent(M), 16);
    assert!(!level1.is_final());

    let final_space = level1.divide(); // 4×4 consumed
    assert_eq!(final_space.extent(M), 4);
    assert_eq!(final_space.extent(N), 4);
    assert!(final_space.is_final());

    // `final_space()` shortcuts straight to the bottom of the stack.
    assert_eq!(space.final_space().extent(M), 4);
    assert!(space.final_space().is_final());
}

// ---- Tiling::over -----------------------------------------------------------

/// Distributing work is a statement about the level's cuts, so it is available wherever cuts are
/// collected.
#[test]
fn over_distributes_work() {
    let space = Tiling::over(&[(M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(
                cubes(CubeAxis::X).instances(5),
                &[(M, 16), (N, 32), (K, 16)],
            );
        })
        .level(|l| {
            l.walk(&[(M, 16), (N, 32), (K, 4)]);
        })
        .build();

    // The shares ride the cubes even though no axis does.
    assert!(matches!(space.cube_count(), CubeCount::Static(5, 1, 1)));
    assert_eq!(space.partitioner().depth(), 2);
}

/// One region each is a box of the grid however many axes are named, so the line deals a dial per
/// axis and states no work. What lets a whole group of batch axes be one `distribute` line
/// without turning the level into a share.
#[test]
fn distributing_several_axes_one_region_each_deals_a_dial_each() {
    let level = |l: &mut cubek_tile::LevelCuts| {
        l.walk(&[(M, 16), (N, 32), (K, 16)]);
    };
    let one_line = Tiling::over(&[(B0, 2), (B1, 3), (M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(cubes(CubeAxis::Z), &[(B0, 1), (B1, 1)]);
            level(l);
        })
        .build();
    let a_dial_each = Tiling::over(&[(B0, 2), (B1, 3), (M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(cubes(CubeAxis::Z), &[(B0, 1)])
                .distribute(cubes(CubeAxis::Z), &[(B1, 1)]);
            level(l);
        })
        .build();

    assert_eq!(one_line, a_dial_each);
    // No work: the walk under it is the one an undistributed level has, and the lowering that
    // reads this is the one that picks the per-region accumulator nest.
    assert!(one_line.partitioner().work().is_none());
    // Both axes still ride Z, one cube per (B0, B1) pair.
    assert!(matches!(one_line.cube_count(), CubeCount::Static(1, 1, 6)));
}

/// The same axes with a count stated cannot be a box: a share begins inside one region and ends
/// inside another, so they are read as one index instead.
#[test]
fn distributing_several_axes_with_a_count_reads_them_as_one_index() {
    let space = Tiling::over(&[(M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(
                cubes(CubeAxis::X).instances(5),
                &[(M, 16), (N, 32), (K, 16)],
            );
        })
        .build();
    assert!(space.partitioner().work().is_some());
    // The shares ride the cubes, and no axis of them does: five cubes, not `4 * 2 * 1`.
    assert!(matches!(space.cube_count(), CubeCount::Static(5, 1, 1)));
}

/// One axis is a box whatever the count, so it is dealt a dial: `instances` there sizes the
/// axis's own tiles across the scope, which is what a cut has always meant.
#[test]
fn distributing_one_axis_with_a_count_is_a_dial() {
    let space = Tiling::over(&[(M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(cubes(CubeAxis::X).instances(4), &[(M, 16)])
                .walk(&[(N, 32), (K, 16)]);
        })
        .build();
    assert!(space.partitioner().work().is_none());
    assert!(matches!(space.cube_count(), CubeCount::Static(4, 1, 1)));
}

/// Nothing named is nothing said: a matmul with no batch axis passes an empty list and the level
/// reads as if the line were not there.
#[test]
fn distributing_no_axis_states_nothing() {
    let space = Tiling::over(&[(M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(cubes(CubeAxis::Z), &[])
                .walk(&[(M, 16), (N, 32), (K, 16)]);
        })
        .build();
    assert!(space.partitioner().work().is_none());
    assert!(matches!(space.cube_count(), CubeCount::Static(1, 1, 1)));
}

/// The plane's lanes combine in registers, which needs them in lockstep. Lanes holding different
/// shares are on different regions, so they never reach a reduction together.
#[test]
#[should_panic = "combine in registers"]
fn distributing_work_across_lanes_is_refused() {
    Tiling::over(&[(M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(lanes(4), &[(M, 16), (N, 32), (K, 16)]);
        })
        .build();
}

/// A share is walked as a nest, one region at a time, so its steps have to be consecutive.
/// Instances taking turns would put a different region under the accumulator at every step.
#[test]
#[should_panic = "instances taking turns would leave no region long enough"]
fn distributing_work_in_turns_is_refused() {
    Tiling::over(&[(M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.distribute(
                cubes(CubeAxis::X).instances(5).interleaved(),
                &[(M, 16), (N, 32), (K, 16)],
            );
        })
        .build();
}

/// A level states each of its axes once, whichever way it states them.
#[test]
#[should_panic = "a level states each of its axes once"]
fn an_axis_both_cut_and_distributed_is_refused() {
    Tiling::over(&[(M, 64), (N, 64), (K, 16)])
        .level(|l| {
            l.walk(&[(K, 16)]).distribute(
                cubes(CubeAxis::X).instances(5),
                &[(M, 16), (N, 32), (K, 16)],
            );
        })
        .build();
}

// ---- A level that cuts nothing --------------------------------------------

/// A level whose edges are the extents handed to it has one instance on every axis, so it
/// partitions nothing. It stays all the same: the kernel walks the levels it stated, one loop
/// per level, so the space has to hold every one of them. A one-region walk folds away in the
/// kernel, so keeping it costs nothing.
#[test]
fn a_level_that_cuts_nothing_is_kept() {
    let plain = Tiling::over(&[(M, 64), (N, 64)])
        .level(|l| {
            l.walk(&[(M, 16), (N, 32)]);
        })
        .build();
    let space = Tiling::over(&[(M, 64), (N, 64)])
        .level(|l| {
            l.walk(&[(M, 16), (N, 32)]);
        })
        // The same edges again: nothing left to cut, still a level.
        .level(|l| {
            l.walk(&[(M, 16), (N, 32)]);
        })
        .build();

    assert_ne!(space, plain);
    assert_eq!(space.partitioner().depth(), 2);
    assert_eq!(space.divide().extent(M), 16);
    assert_eq!(space.divide().divide().extent(M), 16);
    assert!(space.divide().divide().is_final());
}

/// The only level of a space stays even when its cuts take the whole extent: a space with a
/// level is a tile walked in cells, and a space with none is the cell. A one-region walk is the
/// degenerate case of the first, not the second: the shape a plan takes when the knob it splits
/// on (attention's split count, a plane grid of one) lands on 1.
#[test]
fn the_only_level_stays_even_when_it_cuts_nothing() {
    let space = Tiling::over(&[(M, 64), (N, 64)])
        .level(|l| {
            l.walk(&[(M, 64), (N, 64)]);
        })
        .build();

    assert_eq!(space.partitioner().depth(), 1);
    assert!(space.divide().is_final());
}
