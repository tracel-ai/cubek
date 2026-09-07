//! Unit tests for [`Space`]

use cubecl::prelude::*;
use cubek_tile::{Axis, Cut, KernelForm, Launcher, Level, Space};

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

// ---- Level::child (the tiling scheme) --------------------------------------

fn sequential(edges: &[(Axis, usize)]) -> Level {
    Level::walk(edges)
}

#[test]
fn a_level_cuts_each_axis_to_its_edge() {
    let level = sequential(&[(M, 4), (N, 3), (K, 2)]);
    let tile = level.child(&Space::new(&[(M, 16), (N, 12), (K, 8)]));
    assert_eq!(tile.extent(M), 4);
    assert_eq!(tile.extent(N), 3);
    assert_eq!(tile.extent(K), 2);
    assert_eq!(tile.rank(), 3);
}

#[test]
fn levels_chain_into_a_multi_level_scheme() {
    let space = Space::new(&[(M, 64), (N, 64)]);
    let level1 = sequential(&[(M, 16), (N, 16)]).child(&space);
    let level2 = sequential(&[(M, 4), (N, 4)]).child(&level1);

    assert_eq!(level1.extent(M), 16);
    assert_eq!(level2.extent(M), 4);
    assert_eq!(level2.extent(N), 4);
    assert_eq!(
        space.leaf(&[
            sequential(&[(M, 16), (N, 16)]),
            sequential(&[(M, 4), (N, 4)])
        ]),
        level2
    );
}

// ---- overhangs -------------------------------------------------------------

/// A cpu_gemm-shaped two-level scheme: a cube tile of `planes × leaf` leaves over `(m, n, k)`,
/// K cut to its full extent at the cube level (sequential contraction) then to `leaf_k`.
fn cpu_gemm_nest(m: usize, n: usize, k: usize) -> Launcher {
    let (leaf_m, leaf_n, leaf_k) = (8, 8, 4);
    let (planes_m, planes_n) = (2, 4);
    Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, m), (N, n), (K, k)]),
        vec![
            sequential(&[(M, planes_m * leaf_m), (N, planes_n * leaf_n), (K, k)]),
            sequential(&[(M, leaf_m), (N, leaf_n), (K, leaf_k)]),
        ],
        KernelForm::Static,
    )
}

fn hangs(launcher: &Launcher, axis: Axis) -> bool {
    launcher.space().overhangs(launcher.levels(), axis)
}

#[test]
fn overhangs_matches_cpu_gemm_checks() {
    // Every level divides: cube tiles 16×32, leaves 8×8×4.
    let launcher = cpu_gemm_nest(64, 64, 16);
    assert!(!hangs(&launcher, M));
    assert!(!hangs(&launcher, N));
    assert!(!hangs(&launcher, K));

    // m = 40 is not a multiple of the cube tile (16): M overhangs (cpu_gemm's check_m).
    // Within a cube the plane split is exact, so the leaf level adds nothing.
    assert!(hangs(&cpu_gemm_nest(40, 64, 16), M));

    // K's cube-level cut is its full extent (always divides); k = 18 fails only at the
    // leaf (leaf_k = 4): the deeper level alone drives the overhang (cpu_gemm's check_k).
    let launcher = cpu_gemm_nest(64, 64, 18);
    assert!(hangs(&launcher, K));
    assert!(!hangs(&launcher, M));
}

#[test]
fn overhangs_when_a_deeper_edge_misdivides_its_parent() {
    // Top divides (32 % 16 == 0) but the second edge doesn't divide the first (16 % 3 != 0):
    // the parent edge, not the top extent, is what each level must divide.
    let launcher = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, 32)]),
        vec![sequential(&[(M, 16)]), sequential(&[(M, 3)])],
        KernelForm::Static,
    );
    assert!(hangs(&launcher, M));
}

#[test]
fn overhangs_with_no_level_never() {
    // No level: nothing to misdivide.
    assert!(!Space::new(&[(M, 7)]).overhangs(&[], M));
}

#[test]
#[should_panic(expected = "concrete space")]
fn overhangs_dynamic_axis_panics() {
    let launcher = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, 64)]).all_dynamic(),
        vec![sequential(&[(M, 16)])],
        KernelForm::Static,
    );
    let _ = hangs(&launcher, M);
}

// ---- Level constructors ----------------------------------------------------------

/// The tiles of several axes dealt as one index: the shares ride the cubes even though no axis
/// does, so the launch grid is their count.
#[test]
fn shared_tiles_launch_their_instances() {
    let launcher = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, 64), (N, 64), (K, 16)]),
        vec![
            Level::cubes(&[(M, 16), (N, 32), (K, 16)]).shared_by(5),
            Level::walk(&[(M, 16), (N, 32), (K, 4)]),
        ],
        KernelForm::Static,
    );
    assert!(launcher.level(0).work().is_some());
    // Five cubes, not `4 * 2 * 1`.
    assert!(matches!(launcher.cube_count(), CubeCount::Static(5, 1, 1)));
    assert_eq!(launcher.levels().len(), 2);
}

/// Batch axes ride `Z` one tile each, however many there are and however they are listed: a
/// box of the grid, not a share.
#[test]
fn batches_are_a_dial_each() {
    let one_line = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(B0, 2), (B1, 3), (M, 64), (N, 64), (K, 16)]),
        vec![Level::cubes(&[(M, 16), (N, 32)]).batches(&[B0, B1])],
        KernelForm::Static,
    );
    let a_dial_each = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(B0, 2), (B1, 3), (M, 64), (N, 64), (K, 16)]),
        vec![
            Level::cubes(&[(M, 16), (N, 32)])
                .batches(&[B0])
                .batches(&[B1]),
        ],
        KernelForm::Static,
    );

    assert_eq!(one_line.levels(), a_dial_each.levels());
    // No work: the lowering that reads this is the one that picks the per-region accumulator
    // nest.
    assert!(one_line.level(0).work().is_none());
    // Both axes ride Z, one cube per (B0, B1) pair, behind the `4 x 2` grid on X and Y.
    assert!(matches!(one_line.cube_count(), CubeCount::Static(4, 2, 6)));
}

/// One axis is a box whatever the count, so `across` on it sizes the axis's own tiles over the
/// scope, which is what a cut has always meant: no work is stated.
#[test]
fn one_axis_across_a_count_is_a_dial() {
    let launcher = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, 64), (N, 64), (K, 16)]),
        vec![
            Level::cubes(&[Cut::new(M, 16).across(4)]),
            Level::walk(&[(N, 32)]),
        ],
        KernelForm::Static,
    );
    assert!(launcher.level(0).work().is_none());
    assert!(matches!(launcher.cube_count(), CubeCount::Static(4, 1, 1)));
}

/// Nothing named is nothing said: a level that names no axis cuts every cube the whole space.
#[test]
fn a_level_naming_no_axis_deals_everything_to_one_cube() {
    let launcher = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, 64), (N, 64), (K, 16)]),
        vec![
            Level::cubes::<Cut>(&[]),
            Level::walk(&[(M, 16), (N, 32), (K, 16)]),
        ],
        KernelForm::Static,
    );
    assert!(launcher.level(0).work().is_none());
    assert!(matches!(launcher.cube_count(), CubeCount::Static(1, 1, 1)));
    assert_eq!(&launcher.level(0).child(launcher.space()), launcher.space());
}

/// The plane's lanes combine in registers, which needs them in lockstep. Lanes holding different
/// shares are on different regions, so they never reach a reduction together.
#[test]
#[should_panic = "combine in registers"]
fn sharing_tiles_across_lanes_is_refused() {
    let _ = Level::lanes(&[
        Cut::new(M, 16).across(4),
        Cut::new(N, 32).across(4),
        Cut::new(K, 16).across(4),
    ])
    .shared_by(4);
}

/// A share is a run of one index, so its entries say nothing of their own: a count or a spread on
/// one of them would size a box a share has no use for.
#[test]
#[should_panic = "states a count or a spread of its own"]
fn sharing_tiles_with_a_knob_on_an_entry_is_refused() {
    let _ = Level::cubes(&[
        Cut::new(M, 16).interleaved(),
        Cut::new(N, 32),
        Cut::new(K, 16),
    ])
    .shared_by(5);
}

/// A level states each of its axes once, whichever way it states them.
#[test]
#[should_panic = "a level states each of its axes once"]
fn an_axis_named_twice_is_refused() {
    let _ = Level::cubes(&[(M, 16), (M, 32)]);
}

/// Lanes carve one plane between them, so every entry says how many it takes.
#[test]
#[should_panic = "states no lane count"]
fn lanes_without_a_count_are_refused() {
    let _ = Level::lanes(&[Cut::new(M, 16)]);
}

// ---- A level that cuts nothing --------------------------------------------

/// A level whose edges are the extents handed to it has one instance on every axis, so it
/// partitions nothing. It stays all the same: the kernel walks the levels it stated, one loop
/// per level, so the list has to hold every one of them. A one-region walk folds away in the
/// kernel, so keeping it costs nothing.
#[test]
fn a_level_that_cuts_nothing_is_kept() {
    let plain = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, 64), (N, 64)]),
        vec![Level::walk(&[(M, 16), (N, 32)])],
        KernelForm::Static,
    );
    let launcher = Launcher::implied(
        &cubecl::test_device().client(),
        Space::new(&[(M, 64), (N, 64)]),
        vec![
            Level::walk(&[(M, 16), (N, 32)]),
            // The same edges again: nothing left to cut, still a level.
            Level::walk(&[(M, 16), (N, 32)]),
        ],
        KernelForm::Static,
    );

    assert_ne!(launcher.levels(), plain.levels());
    assert_eq!(launcher.levels().len(), 2);
    assert_eq!(launcher.level(0).child(launcher.space()).extent(M), 16);
    assert_eq!(launcher.space().leaf(launcher.levels()).extent(M), 16);
}
