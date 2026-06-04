//! Unit tests for [`Extent`] and [`Space`] — the comptime coordinate machinery,
//! in particular composite (squashed-batch) extents and the broadcast merge
//! performed by [`Space::merge`].

use cubek_tile::{Axis, ByAxis, Distribution, Extent, Partitioner, Space};

// Matmul-style axis labels reused across the cases below.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B: Axis = Axis(3);

// ---- Extent ---------------------------------------------------------------

#[test]
fn scalar_extent_is_plain() {
    let e = Extent::scalar(12);
    assert_eq!(e.total(), 12);
    assert_eq!(e.len(), 1);
    assert_eq!(e.level(0), 12);
    assert!(!e.is_composite());
}

#[test]
fn composite_extent_keeps_levels_and_flattens_to_product() {
    let e = Extent::new(&[4, 3]);
    assert_eq!(e.len(), 2);
    assert!(e.is_composite());
    assert_eq!(e.level(0), 4);
    assert_eq!(e.level(1), 3);
    assert_eq!(e.total(), 12);
}

#[test]
fn single_level_composite_equals_scalar() {
    // A one-level `new` and `scalar` must be indistinguishable (same Eq/Hash),
    // so the zeroed tail normalization holds.
    assert_eq!(Extent::new(&[7]), Extent::scalar(7));
}

#[test]
fn broadcast_merges_ones_per_level() {
    // The case from the design discussion: lhs batch [4,1], rhs batch [1,3].
    let merged = Extent::new(&[4, 1]).broadcast(Extent::new(&[1, 3]));
    assert_eq!(merged, Extent::new(&[4, 3]));
}

#[test]
fn broadcast_equal_extents_is_identity() {
    assert_eq!(
        Extent::new(&[4, 3]).broadcast(Extent::new(&[4, 3])),
        Extent::new(&[4, 3])
    );
    assert_eq!(
        Extent::scalar(8).broadcast(Extent::scalar(8)),
        Extent::scalar(8)
    );
}

#[test]
fn broadcast_one_against_n_takes_n_either_order() {
    assert_eq!(
        Extent::scalar(1).broadcast(Extent::scalar(5)),
        Extent::scalar(5)
    );
    assert_eq!(
        Extent::scalar(5).broadcast(Extent::scalar(1)),
        Extent::scalar(5)
    );
}

#[test]
#[should_panic(expected = "conflicting extents")]
fn broadcast_conflict_panics() {
    // Neither side is 1 and they differ — a genuine shape mismatch.
    let _ = Extent::scalar(4).broadcast(Extent::scalar(3));
}

#[test]
#[should_panic(expected = "differing level counts")]
fn broadcast_differing_level_counts_panics() {
    let _ = Extent::new(&[4, 3]).broadcast(Extent::scalar(12));
}

// ---- Space ----------------------------------------------------------------

#[test]
fn new_builds_plain_axes() {
    let space = Space::new(&[(M, 4), (N, 8), (K, 2)]);
    assert_eq!(space.rank(), 3);
    assert_eq!(space.extent(M), 4);
    assert_eq!(space.extent(N), 8);
    assert_eq!(space.levels(K).len(), 1);
    assert!(!space.levels(K).is_composite());
}

#[test]
fn with_levels_builds_composite_axis() {
    let space = Space::with_levels(&[(B, Extent::new(&[4, 3])), (M, Extent::scalar(16))]);
    // `extent` flattens for callers that only want the walk size...
    assert_eq!(space.extent(B), 12);
    // ...while `levels` exposes the factorization.
    assert!(space.levels(B).is_composite());
    assert_eq!(space.levels(B).level(0), 4);
    assert_eq!(space.levels(B).level(1), 3);
}

#[test]
fn project_preserves_composite_levels() {
    let space = Space::with_levels(&[
        (B, Extent::new(&[4, 3])),
        (M, Extent::scalar(16)),
        (K, Extent::scalar(8)),
    ]);
    let lhs = space.project(&[B, M, K]);
    assert_eq!(lhs.rank(), 3);
    // The composite B must survive the projection, not get flattened to [12].
    assert_eq!(lhs.levels(B), Extent::new(&[4, 3]));
    assert_eq!(lhs.levels(M), Extent::scalar(16));
}

#[test]
fn merge_broadcasts_shared_batch_axis() {
    // lhs batch [4,1], rhs batch [1,3] → output batch [4,3].
    let lhs = Space::with_levels(&[
        (B, Extent::new(&[4, 1])),
        (M, Extent::scalar(16)),
        (K, Extent::scalar(8)),
    ]);
    let rhs = Space::with_levels(&[
        (B, Extent::new(&[1, 3])),
        (K, Extent::scalar(8)),
        (N, Extent::scalar(4)),
    ]);
    let out = Space::with_levels(&[
        (B, Extent::new(&[4, 3])),
        (M, Extent::scalar(16)),
        (N, Extent::scalar(4)),
    ]);

    let merged = Space::merge(&[&out, &lhs, &rhs]);
    assert_eq!(merged.levels(B), Extent::new(&[4, 3]));
    assert_eq!(merged.extent(B), 12);
    // First-appearance order from `out`: B, M, N, then K from lhs.
    assert_eq!(merged.rank(), 4);
    assert_eq!(merged.axis_at(0), B);
    assert_eq!(merged.axis_at(3), K);
}

#[test]
fn merge_omitted_axis_broadcasts_wholesale() {
    // rhs omits B entirely — it broadcasts along all of the batch.
    let lhs = Space::with_levels(&[
        (B, Extent::new(&[4, 3])),
        (M, Extent::scalar(16)),
        (K, Extent::scalar(8)),
    ]);
    let rhs = Space::new(&[(K, 8), (N, 4)]);

    let merged = Space::merge(&[&lhs, &rhs]);
    assert_eq!(merged.levels(B), Extent::new(&[4, 3]));
    assert!(merged.contains(B));
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
fn merge_conflicting_plain_extent_panics() {
    let lhs = Space::new(&[(M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 4), (N, 4)]); // K disagrees: 8 vs 4
    let _ = Space::merge(&[&lhs, &rhs]);
}

// ---- Space::divide (the tiling schema) ------------------------------------

fn sequential(edges: &[(Axis, usize)]) -> Partitioner {
    let dists = edges
        .iter()
        .map(|&(a, _)| (a, Distribution::Sequential))
        .collect::<Vec<_>>();
    Partitioner::row_major(ByAxis::new(edges), ByAxis::new(&dists))
}

#[test]
fn divide_cuts_each_axis_to_its_sub_tile_edge() {
    // The space owns its partitioner, so `divide` takes no argument.
    let space =
        Space::new(&[(M, 16), (N, 12), (K, 8)]).with_partitioner(sequential(&[(M, 4), (N, 3), (K, 2)]));

    // The child schema is the tile *shape* — every axis shrunk to its edge —
    // and carries no position: it's the same whether we're at m=0 or m=12.
    let tile = space.divide();
    assert_eq!(tile.extent(M), 4);
    assert_eq!(tile.extent(N), 3);
    assert_eq!(tile.extent(K), 2);
    assert_eq!(tile.rank(), 3);
}

#[test]
fn divide_chains_into_a_multi_level_schema() {
    // The schema is a tree of spaces: cut, then attach the next level's
    // partitioner and cut the child — independent of any actual coordinate.
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
    // Stacking partitioners builds the whole multi-level schema up front; each
    // `divide` consumes the head level and carries the remaining (finer) stack, so
    // the kernel can descend without re-attaching between levels.
    let space = Space::new(&[(M, 64), (N, 64)])
        .with_partitioner(sequential(&[(M, 16), (N, 16)]))
        .with_partitioner(sequential(&[(M, 4), (N, 4)]));
    assert!(!space.is_leaf());

    let level1 = space.divide(); // head (16×16) consumed, 4×4 remains
    assert_eq!(level1.extent(M), 16);
    assert!(!level1.is_leaf());

    let leaf = level1.divide(); // 4×4 consumed
    assert_eq!(leaf.extent(M), 4);
    assert_eq!(leaf.extent(N), 4);
    assert!(leaf.is_leaf());

    // `leaf()` shortcuts straight to the bottom of the stack.
    assert_eq!(space.leaf().extent(M), 4);
    assert!(space.leaf().is_leaf());
}

// ---- Extent::sub_extent / composite axes under divide ---------------------

#[test]
fn sub_extent_scalar_is_the_edge() {
    // A plain axis just shrinks to the edge — so all-plain spaces are unchanged.
    assert_eq!(Extent::scalar(16).sub_extent(4), Extent::scalar(4));
    assert_eq!(Extent::scalar(8).sub_extent(8), Extent::scalar(8));
}

#[test]
fn sub_extent_composite_consumes_inner_levels_first() {
    let b = Extent::new(&[4, 3]);
    assert_eq!(b.sub_extent(12), Extent::new(&[4, 3])); // pass-through (full)
    assert_eq!(b.sub_extent(6), Extent::new(&[2, 3])); // two outer × full inner
    assert_eq!(b.sub_extent(3), Extent::new(&[1, 3])); // one outer × full inner
    assert_eq!(b.sub_extent(1), Extent::new(&[1, 1])); // single element
}

#[test]
#[should_panic(expected = "does not align")]
fn sub_extent_misaligned_edge_panics() {
    // 2 spans into the size-3 inner level — no clean row-major sub-block.
    let _ = Extent::new(&[4, 3]).sub_extent(2);
}

#[test]
fn divide_preserves_composite_factorization() {
    // The defect this fixes: a composite (squashed-batch) axis must keep its
    // factorization across a divide, not flatten to a scalar — otherwise deeper
    // levels lose the structure broadcasting needs.
    let space = Space::with_levels(&[(B, Extent::new(&[4, 3])), (M, Extent::scalar(16))])
        .with_partitioner(sequential(&[(B, 6), (M, 4)]));
    let child = space.divide();
    assert_eq!(child.levels(B), Extent::new(&[2, 3]));
    assert!(child.levels(B).is_composite());
    assert_eq!(child.extent(M), 4);
}
