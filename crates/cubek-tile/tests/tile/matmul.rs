//! Matmul as a client of the axis-agnostic [`tile_dsl`] engine: the axis labels
//! `M`, `N`, `K`, the operand roles, the kernel, the matmul lowering
//! ([`mma_gmem`]), and the tests. Over tiles `lhs = {M, K}`, `rhs = {K, N}`,
//! `out = {M, N}` — the lowering is matmul-specific and only uses the DSL's pure
//! tile machinery (`partition`/`copy_from`/`stage_smem`).
#![allow(non_snake_case)]

use cubecl::std::tensor::{AsViewMut, AsViewMutExpand, ViewMut, layout::Coords2d};
use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};

// Glob brings the tile-DSL items *and* the cube-macro-generated `*Expand`
// companions the lowering below needs.
use cubek_tile::*;

// Matmul's axes — the labels this client gives the engine's opaque `Axis`. `B`
// is the leading batch axis; `M`/`N`/`K` are the matrix axes.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B: Axis = Axis(3);

/// Staged matmul on tile-permuted tensors, single cube: every axis is
/// `Sequential`, so the `RowMajor` partitioner walks every output tile in turn.
#[test]
fn matmul_sequential_single_cube() {
    check_matmul(
        8,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (M, Distribution::Sequential),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// One tile per cube: M and N are pinned to 2 cube instances each; with
/// `grid = 2` on both, that's one output tile per cube while K stays sequential.
#[test]
fn matmul_one_tile_per_cube() {
    check_matmul(
        8,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (
                    N,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::Y),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// The same single-cube matmul, `Reversed`: visits output tiles back-to-front,
/// same result.
#[test]
fn matmul_reversed_walk_single_cube() {
    check_matmul(
        8,
        8,
        8,
        Partitioner::reversed(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (M, Distribution::Sequential),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// Contiguous spread sized by `TilesEach`: each cube does 2 contiguous m-tiles,
/// instance count derived (`grid_m / 2 = 2` cubes).
#[test]
fn matmul_contiguous_m_across_cubes() {
    check_matmul(
        16,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// Interleaved spread sized by `Instances`: M split across 2 cubes round-robin
/// (cube 0 → `{0,2}`, cube 1 → `{1,3}`).
#[test]
fn matmul_interleaved_m_across_cubes() {
    check_matmul(
        16,
        8,
        8,
        Partitioner::row_major(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        unit: ComputePrimitive::Cube(CubeDimension::X),
                        spread: Spread::Interleaved,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        ),
    );
}

/// Batched matmul, single cube: a leading `B` axis (one batch per tile, since its
/// sub-tile edge is 1) plus the `{M, N, K}` matrix axes, all `Sequential`. Each
/// operand is `{B, …}`, so `partition`'s trailing-two leaf is the matrix tile and
/// the batch is pinned per walk point. Validates that the same code path matmuls
/// the right 2-D subspace independently per batch.
///
/// `batch_edge` is the batch sub-tile size — a pure layout decision: `1` walks
/// one batch per tile, `b` puts the whole batch *inside* one leaf (the matmul
/// then iterates it in-leaf), and anything between splits the two. All three are
/// the same kernel; only the partitioner changes.
#[test]
fn matmul_batched_walked() {
    check_matmul_batched(3, 8, 8, 8, 4, 1);
}

#[test]
fn matmul_batched_in_leaf() {
    check_matmul_batched(4, 8, 8, 8, 4, 4);
}

#[test]
fn matmul_batched_split() {
    check_matmul_batched(4, 8, 8, 8, 4, 2);
}

fn check_matmul_batched(
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_edge: usize,
    batch_edge: usize,
) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let dtype = f32::as_type_native_unchecked().storage_type();
    let vector_size = 1;

    let partitioner = Partitioner::row_major(
        ByAxis::new(&[
            (B, batch_edge),
            (M, tile_edge),
            (N, tile_edge),
            (K, tile_edge),
        ]),
        ByAxis::new(&[
            (B, Distribution::Sequential),
            (M, Distribution::Sequential),
            (N, Distribution::Sequential),
            (K, Distribution::Sequential),
        ]),
    );

    let space = Space::new(&[(B, b), (M, m), (N, n), (K, k)]);
    let a = TileInput::builder(&client, space.select(&[B, M, K]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let rhs = TileInput::builder(&client, space.select(&[B, K, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.select(&[B, M, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .zeros();

    let cube_count = cube_count_for(&partitioner, &space);
    let cube_dim = CubeDim::new_single();

    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        TileLaunch::new(
            a.view(),
            partitioner.launch(),
            a.space(),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            rhs.view(),
            partitioner.launch(),
            rhs.space(),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            c.view(),
            partitioner.launch(),
            c.space(),
            TileKind::GmemWhole,
        ),
        dtype,
        vector_size,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    // Each operand is a physical-order arange over its `[grid_b, grid…, batch_edge,
    // tile, tile]` buffer, so the value at logical `(batch, i, j)` is its flat
    // physical index. Build the expected batched matmul in that same order.
    let (grid_m, grid_n, grid_k) = (m / tile_edge, n / tile_edge, k / tile_edge);
    let e = batch_edge;
    let lhs_at = |bb: usize, i: usize, kk: usize| -> f32 {
        let (gb, tb) = (bb / e, bb % e);
        let (gi, ti) = (i / tile_edge, i % tile_edge);
        let (gk, tk) = (kk / tile_edge, kk % tile_edge);
        (((((gb * grid_m + gi) * grid_k + gk) * e + tb) * tile_edge + ti) * tile_edge + tk) as f32
    };
    let rhs_at = |bb: usize, kk: usize, j: usize| -> f32 {
        let (gb, tb) = (bb / e, bb % e);
        let (gk, tk) = (kk / tile_edge, kk % tile_edge);
        let (gj, tj) = (j / tile_edge, j % tile_edge);
        (((((gb * grid_k + gk) * grid_n + gj) * e + tb) * tile_edge + tk) * tile_edge + tj) as f32
    };

    let mut expected = vec![0.0f32; b * m * n];
    for bb in 0..b {
        let (gb, tb) = (bb / e, bb % e);
        for gm in 0..grid_m {
            for gn in 0..grid_n {
                for tm in 0..tile_edge {
                    for tn in 0..tile_edge {
                        let (i, j) = (gm * tile_edge + tm, gn * tile_edge + tn);
                        let value = (0..k)
                            .map(|kk| lhs_at(bb, i, kk) * rhs_at(bb, kk, j))
                            .sum::<f32>();
                        let offset = ((((gb * grid_m + gm) * grid_n + gn) * e + tb) * tile_edge
                            + tm)
                            * tile_edge
                            + tn;
                        expected[offset] = value;
                    }
                }
            }
        }
    }
    let (_, expected) = TestInput::builder(
        client,
        shape![b / e, grid_m, grid_n, e, tile_edge, tile_edge],
    )
    .custom(expected)
    .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Drives `launch_staged_matmul` for `C = A @ B` under an arbitrary
/// `partitioner`; the launch geometry is derived from it via [`cube_count_for`].
fn check_matmul(m: usize, n: usize, k: usize, partitioner: Partitioner) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let tile_edge = partitioner.sub_tile_edge(M) as usize;

    let dtype = f32::as_type_native_unchecked().storage_type();
    let vector_size = 1;

    let space = Space::new(&[(M, m), (N, n), (K, k)]);
    let a = TileInput::builder(&client, space.select(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.select(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.select(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .zeros();

    let cube_count = cube_count_for(&partitioner, &space);
    let cube_dim = CubeDim::new_single();

    // The whole matmul is `mma_gmem(c, a, b)` over the launched tiles.
    launch_staged_matmul::launch::<TestRuntime>(
        &client,
        cube_count,
        cube_dim,
        TileLaunch::new(
            a.view(),
            partitioner.launch(),
            a.space(),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            b.view(),
            partitioner.launch(),
            b.space(),
            TileKind::GmemWhole,
        ),
        TileLaunch::new(
            c.view(),
            partitioner.launch(),
            c.space(),
            TileKind::GmemWhole,
        ),
        dtype,
        vector_size,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);

    // Inputs are physical-order aranges over their `[grid, grid, tile, tile]`
    // buffers, so the value the kernel reads at logical `(i, j)` is the element's
    // flat physical index. Build the expected matmul in that same physical order.
    let at = |i: usize, j: usize, cols: usize| -> f32 {
        let grid_c = cols / tile_edge;
        let (gi, ti) = (i / tile_edge, i % tile_edge);
        let (gj, tj) = (j / tile_edge, j % tile_edge);
        (((gi * grid_c + gj) * tile_edge + ti) * tile_edge + tj) as f32
    };
    let (grid_m, grid_n) = (m / tile_edge, n / tile_edge);
    let mut expected = vec![0.0f32; m * n];
    for gm in 0..grid_m {
        for gn in 0..grid_n {
            for tm in 0..tile_edge {
                for tn in 0..tile_edge {
                    let (i, j) = (gm * tile_edge + tm, gn * tile_edge + tn);
                    let value = (0..k).map(|kk| at(i, kk, k) * at(kk, j, n)).sum::<f32>();
                    let offset = ((gm * grid_n + gn) * tile_edge + tm) * tile_edge + tn;
                    expected[offset] = value;
                }
            }
        }
    }
    let (_, expected) = TestInput::builder(
        client,
        shape![m / tile_edge, n / tile_edge, tile_edge, tile_edge],
    )
    .custom(expected)
    .generate_with_f32_host_data();

    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The kernel: every operand is a [`Tile`] (a semantic view + its space +
/// partitioner), so the whole matmul is one line.
#[cube(launch)]
fn launch_staged_matmul<E: Numeric, S: Size>(
    a: &Tile<'_, E, S>,
    b: &Tile<'_, E, S>,
    c: &mut Tile<'_, E, S>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    mma_gmem::<E, S>(c, a, b);
}

// ---------------------------------------------------------------------------
// The matmul lowering. This is matmul-specific (it knows the operand roles and
// the scalar contraction), so it lives with the client, not in the tile DSL —
// it only uses the DSL's pure tile machinery (partition / copy_from / stage_smem).
// ---------------------------------------------------------------------------

/// Accumulator in global memory. Walks the partitioner; each step stages both
/// operand leaves into shared memory and accumulates the product into the output
/// leaf.
#[cube]
fn mma_gmem<E: Numeric, S: Size>(
    out: &mut Tile<'_, E, S>,
    lhs: &Tile<'_, E, S>,
    rhs: &Tile<'_, E, S>,
) {
    // The accumulator is written through its (interior-mutable) views, so a
    // shared reborrow is all the lowering needs.
    let out = &*out;

    // The operation ranges over the union of its operands' spaces
    // ({M,N} ∪ {M,K} ∪ {K,N} = {M,N,K}, or with a leading batch axis B) and
    // contracts the axes the output drops.
    let space = comptime!(Space::union(&[&out.space, &lhs.space, &rhs.space]));
    let contracted = comptime!(space.contracting(&out.space));
    comptime!(assert!(
        !contracted.is_empty(),
        "mma: the output must drop at least one (contracted) axis"
    ));

    // A shared-memory twin of each operand's whole leaf — addressed by `matrix`
    // exactly like the global leaf, so staging is a single `copy_from`.
    let mut a_smem = Shared::<[Vector<E, S>]>::new_slice(
        lhs.partitioner.leaf_size(comptime!(lhs.space.clone())),
    );
    let mut b_smem = Shared::<[Vector<E, S>]>::new_slice(
        rhs.partitioner.leaf_size(comptime!(rhs.space.clone())),
    );
    let mut a_tile = stage_smem::<E, S>(
        a_smem.view_mut(smem_layout(comptime!(lhs.space.clone()), &lhs.partitioner)),
        comptime!(lhs.space.clone()),
        lhs.partitioner.clone(),
    );
    let mut b_tile = stage_smem::<E, S>(
        b_smem.view_mut(smem_layout(comptime!(rhs.space.clone()), &rhs.partitioner)),
        comptime!(rhs.space.clone()),
        rhs.partitioner.clone(),
    );

    // The matmul's tile grid (gathered from the operands), walked by the
    // partitioner.
    let grid = mma_grid::<E, S>(out, lhs, rhs, comptime!(space.clone()));
    let walk = out.partitioner.walk(grid);
    let total = walk.total();
    for i in 0..total {
        let point = walk.point(i);

        let a_leaf = lhs.partition(&point);
        let b_leaf = rhs.partition(&point);
        let acc_leaf = out.partition(&point);

        // Stage both operand leaves into shared memory (the `copy_from` pillar),
        // then contract each 2-D matrix of the leaf (more than one when the batch
        // axis is sub-tiled wider than 1).
        a_tile.copy_from(&a_leaf);
        b_tile.copy_from(&b_leaf);
        let matrices = acc_leaf.matrices();
        for j in 0..matrices {
            let mut acc = acc_leaf.matrix(j);
            mma_smem::<E, S>(&mut acc.view, &a_tile.matrix(j).view, &b_tile.matrix(j).view);
        }
    }
}

/// This matmul's tile [`Grid`] for `space`: each axis's tile count read from an
/// operand that carries it. The partitioner takes the grid from here.
#[cube]
fn mma_grid<E: Numeric, S: Size>(
    out: &Tile<'_, E, S>,
    lhs: &Tile<'_, E, S>,
    rhs: &Tile<'_, E, S>,
    #[comptime] space: Space,
) -> Grid {
    let mut counts = Sequence::<usize>::new();
    #[unroll]
    for p in 0..comptime!(space.rank()) {
        counts.push(tiles_of::<E, S>(out, lhs, rhs, comptime!(space.axis_at(p))));
    }
    Grid::new(counts, space)
}

/// The runtime tile count along `axis`, read from whichever operand carries it.
/// Every union axis is in at least one operand.
#[cube]
fn tiles_of<E: Numeric, S: Size>(
    out: &Tile<'_, E, S>,
    lhs: &Tile<'_, E, S>,
    rhs: &Tile<'_, E, S>,
    #[comptime] axis: Axis,
) -> usize {
    if comptime!(out.space.contains(axis)) {
        out.tiles(axis)
    } else if comptime!(lhs.space.contains(axis)) {
        lhs.tiles(axis)
    } else {
        rhs.tiles(axis)
    }
}

/// Scalar 2-D contraction `acc(i, j) += Σ_c lhs(i, c) · rhs(c, j)`, shapes read
/// from the views.
#[cube]
fn mma_smem<E: Numeric, S: Size>(
    acc: &mut ViewMut<'_, Vector<E, S>, Coords2d>,
    lhs: &ViewMut<'_, Vector<E, S>, Coords2d>,
    rhs: &ViewMut<'_, Vector<E, S>, Coords2d>,
) {
    let (m, k) = lhs.shape();
    let (_, n) = rhs.shape();

    for i in 0..m {
        for j in 0..n {
            let mut value = acc.read((i, j));
            for c in 0..k {
                value += lhs.read((i, c)) * rhs.read((c, j));
            }
            acc.write((i, j), value);
        }
    }
}
