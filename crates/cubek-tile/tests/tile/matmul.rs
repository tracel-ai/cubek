//! Matmul as a client of the axis-agnostic [`tile_dsl`] engine: the axis labels
//! `M`, `N`, `K`, the operand roles, the kernel, the matmul lowering
//! ([`mma_gmem`]), and the tests. Over tiles `lhs = {M, K}`, `rhs = {K, N}`,
//! `out = {M, N}` — the lowering is matmul-specific and only uses the DSL's pure
//! tile machinery (`partition`/`copy_from`/`stage_smem`).
#![allow(non_snake_case)]

use cubecl::std::tensor::{
    AsViewMut, AsViewMutExpand, ViewMut,
    layout::{Coords2d, CoordsDyn},
};
use cubecl::{TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};

// Glob brings the tile-DSL items *and* the cube-macro-generated `*Expand`
// companions the lowering below needs.
use cubek_tile::*;

// Matmul's three axes — the labels this client gives the engine's opaque `Axis`.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

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

    // The whole matmul is `mma_gmem(&c, &a, &b)` over the launched tiles.
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
    a: Tile<'_, E, S, CoordsDyn>,
    b: Tile<'_, E, S, CoordsDyn>,
    c: Tile<'_, E, S, CoordsDyn>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    mma_gmem::<E, S>(&c, &a, &b);
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
    out: &Tile<'_, E, S, CoordsDyn>,
    lhs: &Tile<'_, E, S, CoordsDyn>,
    rhs: &Tile<'_, E, S, CoordsDyn>,
) {
    // The operation ranges over the union of its operands' spaces
    // ({M,N} ∪ {M,K} ∪ {K,N} = {M,N,K}) and contracts the axes the output drops.
    let space = comptime!(Space::union(&[&out.space, &lhs.space, &rhs.space]));
    let contracted = comptime!(space.contracting(&out.space));
    comptime!(assert!(
        !contracted.is_empty(),
        "mma: the output must drop at least one (contracted) axis"
    ));

    // Stage each operand at its own axes' sub-tile size (comptime).
    let a_rows = lhs
        .partitioner
        .sub_tile_edge(comptime!(lhs.space.axis_at(0)));
    let a_cols = lhs
        .partitioner
        .sub_tile_edge(comptime!(lhs.space.axis_at(1)));
    let b_rows = rhs
        .partitioner
        .sub_tile_edge(comptime!(rhs.space.axis_at(0)));
    let b_cols = rhs
        .partitioner
        .sub_tile_edge(comptime!(rhs.space.axis_at(1)));

    let mut a_smem = Shared::<[Vector<E, S>]>::new_slice(comptime!((a_rows * a_cols) as usize));
    let mut b_smem = Shared::<[Vector<E, S>]>::new_slice(comptime!((b_rows * b_cols) as usize));
    let mut a_tile = stage_smem::<E, S>(
        a_smem.view_mut(smem_tile_layout(a_rows, a_cols)),
        comptime!(lhs.space.clone()),
        lhs.partitioner.clone(),
    );
    let mut b_tile = stage_smem::<E, S>(
        b_smem.view_mut(smem_tile_layout(b_rows, b_cols)),
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
        let mut acc = out.partition(&point);

        a_tile.copy_from(&a_leaf);
        b_tile.copy_from(&b_leaf);
        mma_smem::<E, S>(&mut acc.view, &a_tile.view, &b_tile.view);
    }
}

/// This matmul's tile [`Grid`] for `space`: each axis's tile count read from an
/// operand that carries it. The partitioner takes the grid from here.
#[cube]
fn mma_grid<E: Numeric, S: Size>(
    out: &Tile<'_, E, S, CoordsDyn>,
    lhs: &Tile<'_, E, S, CoordsDyn>,
    rhs: &Tile<'_, E, S, CoordsDyn>,
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
    out: &Tile<'_, E, S, CoordsDyn>,
    lhs: &Tile<'_, E, S, CoordsDyn>,
    rhs: &Tile<'_, E, S, CoordsDyn>,
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
