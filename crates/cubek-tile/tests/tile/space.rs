//! Unit tests for [`Space`]

use cubek_tile::{Axis, ByAxis, Distribution, Partitioner, Space};

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
    Partitioner::row_major(ByAxis::new(edges), ByAxis::new(&dists)).staged()
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
