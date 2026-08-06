//! Gather-reduce: contractions over more than one abstract axis.
//!
//! The first client is the simplest one that cannot be a matmul: a matmul whose `K` is declared as
//! *two* axes. Nothing about the data or the arithmetic changes, only how many axes the operands
//! contract, so the answer must be exactly the plain matmul's. That isolates the multi-axis reduce
//! nest from the projection machinery, which the stencil tests exercise separately.
#![allow(non_snake_case)]

use cubecl::{
    Runtime, TestRuntime,
    prelude::*,
    zspace::{Shape, shape},
};
use cubek_test_utils::{HostData, HostDataType, TestInput};

use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
// `K` split in two: `K1` the major half, `K2` the minor (and the axis both operands line along).
const K1: Axis = Axis(3);
const K2: Axis = Axis(4);
const B: Axis = Axis(5);

/// `c.zero(); c.mma(a, b)` over whatever space it is handed: the same body serves the one-axis
/// and the two-axis contraction, which is the point.
#[cube(launch)]
fn gather_matmul_kernel<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: StorageType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

/// Small integers, so every product and partial sum is exact in `f32` and the two kernels can be
/// compared for equality rather than closeness.
fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n).map(|i| (i % period) as f32).collect()
}

/// Run `space` with `a`/`b` declared over `a_axes`/`b_axes` and the given buffer shapes, and
/// return the `m × n` output. The buffers are contiguous either way, so a rank-3 declaration of
/// the same bytes is just a finer reading of the same layout.
fn run(
    a_shape: Shape,
    b_shape: Shape,
    c_shape: Shape,
    a_axes: &[Axis],
    b_axes: &[Axis],
    c_axes: &[Axis],
    space: Space,
) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::as_type_native_unchecked().storage_type();

    let (a_len, b_len) = (a_shape.num_elements(), b_shape.num_elements());

    let (a_handle, _) = TestInput::builder(client.clone(), a_shape)
        .dtype(f32_ty)
        .custom(ramp(a_len, 7))
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), b_shape)
        .dtype(f32_ty)
        .custom(ramp(b_len, 5))
        .generate_with_f32_host_data();
    let c_handle = TestInput::builder(client.clone(), c_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    gather_matmul_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            a_handle.binding().into_tensor_arg(),
            TileSpec::direct(a_axes),
        ),
        TileArgLaunch::new(
            b_handle.binding().into_tensor_arg(),
            TileSpec::direct(b_axes),
        ),
        TileArgLaunch::new(
            c_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(c_axes),
        ),
        space,
        f32_ty,
    );

    HostData::from_tensor_handle(&client, c_handle, HostDataType::F32)
}

/// The one-axis reference: an ordinary `{M, N, K}` matmul through the 2-D register leaf.
fn plain(m: usize, n: usize, k: usize, tm: usize, tn: usize) -> HostData {
    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::sequential(tm))
                .axis(N, Cut::sequential(tn))
                .axis(K, Cut::sequential(k))
        })
        .build();
    run(
        shape![m, k],
        shape![k, n],
        shape![m, n],
        &[M, K],
        &[K, N],
        &[M, N],
        space,
    )
}

/// [`plain`] with a leading batch axis both operands span.
fn plain_batched(b: usize, m: usize, n: usize, k: usize, tm: usize, tn: usize) -> HostData {
    let space = Tiling::new()
        .extents(&[(B, b), (M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(B, Cut::sequential(1))
                .axis(M, Cut::sequential(tm))
                .axis(N, Cut::sequential(tn))
                .axis(K, Cut::sequential(k))
        })
        .build();
    run(
        shape![b, m, k],
        shape![b, k, n],
        shape![b, m, n],
        &[B, M, K],
        &[B, K, N],
        &[B, M, N],
        space,
    )
}

fn assert_same(got: &HostData, want: &HostData, dims: &[usize]) {
    let total: usize = dims.iter().product();
    for flat in 0..total {
        let mut idx = vec![0usize; dims.len()];
        let mut rest = flat;
        for p in (0..dims.len()).rev() {
            idx[p] = rest % dims[p];
            rest /= dims[p];
        }
        assert_eq!(
            got.get_f32(&idx),
            want.get_f32(&idx),
            "split-K disagrees with the plain matmul at {idx:?}"
        );
    }
}

/// Both halves of the split reduce ride the leaf: one `mma` call unravels a `k1 × k2` nest.
#[test]
fn split_k_whole_reduce_at_leaf() {
    let (m, n, k1, k2) = (8, 8, 3, 4);
    let (k, tm, tn) = (k1 * k2, 4, 4);

    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K1, k1), (K2, k2)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::sequential(tm))
                .axis(N, Cut::sequential(tn))
                .axis(K1, Cut::sequential(k1))
                .axis(K2, Cut::sequential(k2))
        })
        .build();

    let got = run(
        shape![m, k1, k2],
        shape![k1, k2, n],
        shape![m, n],
        &[M, K1, K2],
        &[K1, K2, N],
        &[M, N],
        space,
    );
    assert_same(&got, &plain(m, n, k, tm, tn), &[m, n]);
}

/// The major half is walked instead: `k1` sequential `mma` calls, each reducing `k2`. The leaf
/// still contracts two axes (`K1` is present at extent 1), so it still takes the gather path.
#[test]
fn split_k_major_half_walked() {
    let (m, n, k1, k2) = (8, 8, 3, 4);
    let (k, tm, tn) = (k1 * k2, 4, 4);

    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K1, k1), (K2, k2)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::sequential(tm))
                .axis(N, Cut::sequential(tn))
                .axis(K1, Cut::sequential(1))
                .axis(K2, Cut::sequential(k2))
        })
        .build();

    let got = run(
        shape![m, k1, k2],
        shape![k1, k2, n],
        shape![m, n],
        &[M, K1, K2],
        &[K1, K2, N],
        &[M, N],
        space,
    );
    assert_same(&got, &plain(m, n, k, tm, tn), &[m, n]);
}

/// A batch axis alongside the two contracted ones: the leaf's batch unravel and its reduce
/// unravel are independent nests, so they must not be confused for one another.
#[test]
fn split_k_with_a_batch_axis() {
    let (b, m, n, k1, k2) = (3, 4, 8, 2, 4);
    let (k, tm, tn) = (k1 * k2, 4, 4);

    let space = Tiling::new()
        .extents(&[(B, b), (M, m), (N, n), (K1, k1), (K2, k2)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(B, Cut::sequential(1))
                .axis(M, Cut::sequential(tm))
                .axis(N, Cut::sequential(tn))
                .axis(K1, Cut::sequential(k1))
                .axis(K2, Cut::sequential(k2))
        })
        .build();

    let got = run(
        shape![b, m, k1, k2],
        shape![b, k1, k2, n],
        shape![b, m, n],
        &[B, M, K1, K2],
        &[B, K1, K2, N],
        &[B, M, N],
        space,
    );
    assert_same(&got, &plain_batched(b, m, n, k, tm, tn), &[b, m, n]);
}
