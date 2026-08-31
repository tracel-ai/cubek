//! Splitting the contraction across cubes, and putting the pieces back together.
//!
//! A cube that does not do the whole contraction holds a slice of every output cell it touches,
//! and the slices have to be combined. This file is the combine that needs nothing new: the split
//! is spelled as an *axis*.
//!
//! `K` is declared as two axes, `(KB, KI)`, addressing one physical `K` through
//! [`PhysicalAxisMap::disjoint`] exactly as a quantization block does. `KB` counts the splits and
//! rides the cubes; `KI` is the position inside one split and is what the contraction walks. The
//! output is bound over `[KB, M, N]`, so it *spans* the axis being split, and the whole thing is
//! a batched matmul whose batch is the split index: no cube shares a cell with any other and
//! nothing is partial. A second pass reduces the `KB` axis away.
//!
//! Two kernels and an extra buffer, but the engine is untouched, which is what makes this the
//! reference the in-kernel combines are measured against: it is the same arithmetic, cut the same
//! way, differing only in where the pieces are added up.
#![allow(non_snake_case)]

use cubecl::{Runtime, TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};

use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
// `K` as two axes: `KB` counts the splits (one per cube), `KI` walks inside one.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// The split contraction: a batched matmul whose batch is the split index, writing one partial
/// per split. `mm` owns the init, so the partials buffer needs no zeroing.
#[cube(launch)]
fn split_partials<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    partials: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut partials = partials.tile(space);
    partials.mm(&a, &b, Semiring::SUM_PROD);
}

/// The second pass: fold the split axis away.
#[cube(launch)]
fn reduce_splits<E: Numeric>(
    partials: &TileArg<'_, E, Const<1>>,
    out: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let partials = partials.tile(comptime!(space.clone()));
    let mut out = out.tile(space);
    out.reduce_axis(&partials, Monoid::Sum);
}

/// The whole pipeline over `(m, n, k)` cut into `splits`: the partials, then the fold. Returns
/// both, since a wrong total and a wrong partial are different bugs.
fn run_split_k(m: usize, n: usize, k: usize, splits: usize) -> (HostData, HostData) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    assert!(k.is_multiple_of(splits), "the test shapes divide evenly");
    let inside = k / splits;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    // Poisoned, not zeroed: `mm` states `c = a·b`, so nothing here may count.
    let partials = TestInput::builder(client.clone(), shape![splits, m, n])
        .dtype(dtype)
        .uniform(4242, 10., 100.)
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .uniform(4243, 10., 100.)
        .generate_without_host_data();

    // One split per cube, the whole output tile in each: the split is the only thing on the grid.
    let split_space = Tiling::new()
        .extents(&[(M, m), (N, n), (KB, splits), (KI, inside)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(m))
                .axis(N, Cut::sequential(n))
                .axis(KB, Cut::cube(CubeAxis::Z, 1))
                .axis(KI, Cut::sequential(inside))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    // `a` is `[M, K]` and `b` is `[K, N]` in memory: one physical `K` dim each, addressed by the
    // two logical axes. `inside` is `KB`'s stride through it, `1` is `KI`'s.
    split_partials::launch::<TestRuntime>(
        &client,
        split_space.cube_count(),
        split_space.cube_dim(&client),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, inside), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, inside), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            partials.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[KB, M, N]),
        ),
        split_space,
        dtype,
    );

    let fold_space = Tiling::new()
        .extents(&[(M, m), (N, n), (KB, splits)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::X, 1))
                .axis(N, Cut::sequential(n))
                .axis(KB, Cut::sequential(splits))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    reduce_splits::launch::<TestRuntime>(
        &client,
        fold_space.cube_count(),
        fold_space.cube_dim(&client),
        TileArgLaunch::new(
            partials.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[KB, M, N]),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        fold_space,
        dtype,
    );

    (
        HostData::from_tensor_handle(&client, partials, HostDataType::F32),
        HostData::from_tensor_handle(&client, out, HostDataType::F32),
    )
}

/// The reference: `a·b` over the whole `K`, and over one split's slice of it.
fn reference(m: usize, n: usize, k: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            rows.clone().map(|p| a(i, p) * b(p, j)).sum()
        })
        .collect()
}

/// Each cube contracts its own slice of `K` and writes it to its own plane of the partials
/// buffer, and the fold adds the planes up. Both halves are checked: a partials buffer that is
/// right and a total that is not would be a broken fold, and the reverse a broken split.
#[test]
fn a_split_contraction_sums_back_to_the_whole() {
    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);
    let inside = k / splits;
    let (partials, out) = run_split_k(m, n, k, splits);

    for s in 0..splits {
        let want = reference(m, n, k, s * inside..(s + 1) * inside);
        for i in 0..m {
            for j in 0..n {
                let have = partials.get_f32(&[s, i, j]);
                let want = want[i * n + j];
                assert!(
                    (have - want).abs() < 1e-3,
                    "split {s} at ({i}, {j}): got {have}, want {want}"
                );
            }
        }
    }

    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = out.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "total at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// A split of one is the unsplit contraction, and has to stay it: the degenerate case is what a
/// selection heuristic falls back to, so it must not be a different program's worth of bugs.
#[test]
fn a_split_of_one_is_the_whole_contraction() {
    let (m, n, k) = (4usize, 4usize, 16usize);
    let (_, out) = run_split_k(m, n, k, 1);

    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = out.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}
