//! Multi-axis reduce: contractions over more than one abstract axis.
//!
//! The first client is the simplest one that cannot be a matmul: a matmul whose `K` is declared as
//! *two* axes. Nothing about the data or the arithmetic changes, only how many axes the operands
//! contract, so the answer must be exactly the plain matmul's. That isolates the multi-axis reduce
//! nest from the projection machinery, which the conv tests exercise separately.
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
fn reduce_matmul_kernel<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    c.mma(&a, &b);
}

/// Seeds `output` for `op` (the identity a fold starts from), then reduces `input` into it.
/// One body serves Sum/Max/Min alike, which is the point: the three kernels below differed only
/// in this seed and the `LeafOp` passed to `reduce_axis`.
#[cube]
fn reduce_body<E: Numeric>(input: &Tile<E>, output: &mut Tile<E>, #[comptime] op: LeafOp) {
    match comptime!(op) {
        LeafOp::Sum => output.zero(),
        LeafOp::Max => output.init(E::min_value()),
        LeafOp::Min => output.init(E::max_value()),
    }
    output.reduce_axis(input, op);
}

#[cube(launch)]
fn reduce_kernel<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] op: LeafOp,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let mut output = output.tile(space);
    reduce_body(&input, &mut output, op);
}

#[cube(launch)]
fn reduce_kernel_v4<E: Numeric>(
    input: &TileArg<'_, E, Const<4>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] op: LeafOp,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let mut output = output.tile(space);
    reduce_body(&input, &mut output, op);
}

/// Reduce an axis-index recipe so a trailing partial tile must be masked without a backing window.
/// `stage` decides whether the recipe is evaluated at the leaf or first materialized into shared
/// memory; either way the source's partial-tile mask must survive, rather than folding values from
/// the padded overhang.
#[cube(launch)]
fn procedural_reduce_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] stage: StagePlan,
    #[define(E)] _dtype: ElemType,
) {
    let input = Tile::<E>::procedural_resident::<AffineCoordinate<E>>(
        comptime!(space.clone()),
        AffineCoordinate::<E> {
            offset: E::new(0.0_f32),
            coefficient: E::new(1.0_f32),
            axis: K,
        },
        stage,
    );
    let mut output = output.tile(space);
    reduce_body(&input, &mut output, comptime!(LeafOp::Max));
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
    let f32_ty = f32::elem_type_native();

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

    reduce_matmul_kernel::launch::<TestRuntime>(
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
        .instruction(Instruction::registers(16), |l| {
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
        .instruction(Instruction::registers(16), |l| {
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
        .instruction(Instruction::registers(16), |l| {
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
/// still contracts two axes (`K1` is present at extent 1), so it still takes the N-D nest.
#[test]
fn split_k_major_half_walked() {
    let (m, n, k1, k2) = (8, 8, 3, 4);
    let (k, tm, tn) = (k1 * k2, 4, 4);

    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K1, k1), (K2, k2)])
        .instruction(Instruction::registers(16), |l| {
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
        .instruction(Instruction::registers(16), |l| {
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

fn run_reduce(
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    space: Space,
    op: LeafOp,
) -> HostData {
    run_reduce_with_vw(in_shape, out_shape, in_axes, out_axes, space, op, 1, &[])
}

/// [`run_reduce`] with the input's per-level [`Residence`] stated, for the walks that stage it.
#[allow(clippy::too_many_arguments)]
fn run_reduce_resident(
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    space: Space,
    op: LeafOp,
    in_residence: &[Residence],
) -> HostData {
    run_reduce_with_vw(
        in_shape,
        out_shape,
        in_axes,
        out_axes,
        space,
        op,
        1,
        in_residence,
    )
}

/// Exercise a 2-D `M × K -> M` reduction and derive the reference fold from `op`, so schedule
/// coverage does not duplicate the three identities and comparison loops.
fn check_2d_reduce(buffering: Buffering, m: usize, k: usize, tm: usize, tk: usize, op: LeafOp) {
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, buffering, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();
    // Every caller of this helper stages: the level is what the buffering coverage exercises.
    let got = run_reduce_resident(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        space,
        op,
        &[Residence::Smem],
    );

    for i in 0..m {
        let values = (0..k).map(|j| ((i * k + j) % 7) as f32);
        let want = match op {
            LeafOp::Sum => values.sum(),
            LeafOp::Max => values.fold(f32::NEG_INFINITY, f32::max),
            LeafOp::Min => values.fold(f32::INFINITY, f32::min),
        };
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "{buffering:?} {op:?} mismatch at index {i}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_reduce_with_vw(
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    space: Space,
    op: LeafOp,
    in_vw: usize,
    in_residence: &[Residence],
) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();
    let in_len = in_shape.num_elements();

    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(ramp(in_len, 7))
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let in_binding = in_handle.binding();
    let out_binding = out_handle.clone().binding();

    match in_vw {
        1 => {
            reduce_kernel::launch::<TestRuntime>(
                &client,
                space.cube_count(),
                space.cube_dim(&client),
                TileArgLaunch::new(
                    in_binding.into_tensor_arg(),
                    TileSpec::direct(in_axes).residence(in_residence),
                ),
                TileArgLaunch::new(out_binding.into_tensor_arg(), TileSpec::direct(out_axes)),
                space,
                op,
                f32_ty,
            );
        }
        4 => {
            reduce_kernel_v4::launch::<TestRuntime>(
                &client,
                space.cube_count(),
                space.cube_dim(&client),
                TileArgLaunch::new(
                    in_binding.into_tensor_arg(),
                    TileSpec::direct(in_axes).residence(in_residence),
                ),
                TileArgLaunch::new(out_binding.into_tensor_arg(), TileSpec::direct(out_axes)),
                space,
                op,
                f32_ty,
            );
        }
        _ => unimplemented!("unsupported in_vw"),
    }

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

#[test]
fn test_reduce_axis_sum_2d_to_1d() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce(shape![m, k], shape![m], &[M, K], &[M], space, LeafOp::Sum);

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(got.get_f32(&[i]), want, "Sum mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_sum_walked_levels() {
    let (m, k, tm, tk) = (8, 16, 4, 4);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce(shape![m, k], shape![m], &[M, K], &[M], space, LeafOp::Sum);

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(got.get_f32(&[i]), want, "Sum mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_max_2d_to_1d() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce(shape![m, k], shape![m], &[M, K], &[M], space, LeafOp::Max);

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got.get_f32(&[i]), want, "Max mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_min_2d_to_1d() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce(shape![m, k], shape![m], &[M, K], &[M], space, LeafOp::Min);

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::INFINITY, f32::min);
        assert_eq!(got.get_f32(&[i]), want, "Min mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_multi_axis_3d_to_1d() {
    let (b, m, k) = (3, 4, 8);
    let space = Tiling::new()
        .extents(&[(B, b), (M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(B, Cut::sequential(1))
                .axis(M, Cut::sequential(2))
                .axis(K, Cut::sequential(4))
        })
        .build();

    let got = run_reduce(
        shape![b, m, k],
        shape![b],
        &[B, M, K],
        &[B],
        space,
        LeafOp::Sum,
    );

    for bi in 0..b {
        let mut want = 0.0f32;
        for mi in 0..m {
            for ki in 0..k {
                let flat = bi * (m * k) + mi * k + ki;
                want += (flat % 7) as f32;
            }
        }
        assert_eq!(got.get_f32(&[bi]), want, "Multi-axis sum mismatch at {bi}");
    }
}

#[test]
fn test_reduce_axis_sum_staged() {
    check_2d_reduce(Buffering::SINGLE, 8, 16, 4, 8, LeafOp::Sum);
}

#[test]
fn test_reduce_axis_sum_double_buffered() {
    check_2d_reduce(Buffering::DOUBLE, 8, 16, 4, 4, LeafOp::Sum);
}

#[test]
fn test_reduce_axis_max_staged() {
    check_2d_reduce(Buffering::SINGLE, 8, 16, 4, 8, LeafOp::Max);
}

#[test]
fn test_reduce_axis_min_staged() {
    check_2d_reduce(Buffering::SINGLE, 8, 16, 4, 8, LeafOp::Min);
}

#[test]
fn test_reduce_axis_max_double_buffered() {
    check_2d_reduce(Buffering::DOUBLE, 8, 16, 4, 4, LeafOp::Max);
}

#[test]
fn test_reduce_axis_min_double_buffered() {
    check_2d_reduce(Buffering::DOUBLE, 8, 16, 4, 4, LeafOp::Min);
}

/// Reduction over an outer axis while retaining the innermost axis (which lines along vector width).
#[test]
fn test_reduce_axis_sum_outer_axis_retained_innermost_v1() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        space,
        LeafOp::Sum,
        1,
        &[],
    );

    for j in 0..k {
        let want: f32 = (0..m).map(|i| ((i * k + j) % 7) as f32).sum();
        assert_eq!(got.get_f32(&[j]), want, "Sum mismatch at index {j}");
    }
}

/// Reduction over an outer axis while retaining the innermost axis with vector_size = 4.
/// Exercises line indexing and lane extraction when the innermost axis is in accumulator space.
#[test]
fn test_reduce_axis_sum_outer_axis_retained_innermost_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        space,
        LeafOp::Sum,
        4,
        &[],
    );

    for j in 0..k {
        let want: f32 = (0..m).map(|i| ((i * k + j) % 7) as f32).sum();
        assert_eq!(
            got.get_f32(&[j]),
            want,
            "Vectorized sum mismatch at index {j}"
        );
    }
}

/// [`run_reduce`], but the input is `checked(true)`: a non-divisible reduced axis leaves an
/// overhang past the operand's real data, and only a checked operand masks it instead of reading
/// garbage.
#[allow(clippy::too_many_arguments)]
fn run_reduce_checked(
    in_data: Vec<f32>,
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    space: Space,
    op: LeafOp,
    in_residence: &[Residence],
) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data)
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let in_binding = in_handle.binding();
    let out_binding = out_handle.clone().binding();

    reduce_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            in_binding.into_tensor_arg(),
            TileSpec::direct(in_axes)
                .checked(true)
                .residence(in_residence),
        ),
        TileArgLaunch::new(out_binding.into_tensor_arg(), TileSpec::direct(out_axes)),
        space,
        op,
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

fn nondivisible_k_space(m: usize, k: usize, tk: usize, schedule: Buffering) -> Space {
    Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, schedule, |l| {
            l.axis(M, Cut::sequential(m)).axis(K, Cut::sequential(tk))
        })
        .build()
}

#[test]
fn test_reduce_axis_sum_nondivisible_k() {
    let (m, k, tk) = (4, 6, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk, Buffering::SINGLE),
        LeafOp::Sum,
        &[],
    );

    for i in 0..m {
        let want: f32 = data[i * k..(i + 1) * k].iter().sum();
        assert_eq!(got.get_f32(&[i]), want, "Sum mismatch at row {i}");
    }
}

#[test]
fn test_reduce_axis_sum_nondivisible_k_staged() {
    let (m, k, tk) = (4, 6, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk, Buffering::SINGLE),
        LeafOp::Sum,
        &[Residence::Smem],
    );

    for i in 0..m {
        let want: f32 = data[i * k..(i + 1) * k].iter().sum();
        assert_eq!(got.get_f32(&[i]), want, "Staged sum mismatch at row {i}");
    }
}

/// The input read where it lies, two slots deep: the ring materializes nothing, so its slots hold
/// windows alone, read at its own region. The depth stays the level's to state, whatever the input
/// costs.
#[test]
fn test_reduce_axis_sum_in_place_double_buffered() {
    let (m, k, tk) = (4, 10, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk, Buffering::DOUBLE),
        LeafOp::Sum,
        &[Residence::InPlace],
    );

    for i in 0..m {
        let want: f32 = data[i * k..(i + 1) * k].iter().sum();
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "In-place double-buffered sum mismatch at row {i}"
        );
    }
}

#[test]
fn test_reduce_axis_sum_nondivisible_k_double_buffered() {
    let (m, k, tk) = (4, 10, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk, Buffering::DOUBLE),
        LeafOp::Sum,
        &[Residence::Smem],
    );

    for i in 0..m {
        let want: f32 = data[i * k..(i + 1) * k].iter().sum();
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Double-buffered sum mismatch at row {i}"
        );
    }
}

#[test]
fn test_reduce_axis_max_nondivisible_k() {
    let (m, k, tk) = (4, 6, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk, Buffering::SINGLE),
        LeafOp::Max,
        &[],
    );

    for i in 0..m {
        let want = data[i * k..(i + 1) * k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got.get_f32(&[i]), want, "Max mismatch at row {i}");
    }
}

/// Max over `k` of the axis-index recipe, with the source staged as `stage` says. The last real
/// `k` is 5, so a mask that leaked the padded overhang would report the tile edge instead.
fn check_procedural_reduce(stage: StagePlan) {
    let (m, k, tk) = (4, 6, 4);
    let space = nondivisible_k_space(m, k, tk, Buffering::SINGLE);
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![m])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    procedural_reduce_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M]),
        ),
        space,
        stage,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..m {
        assert_eq!(got.get_f32(&[row]), 5.0);
    }
}

#[test]
fn test_procedural_max_masks_nondivisible_k() {
    check_procedural_reduce(StagePlan::in_place());
}

#[test]
fn test_staged_procedural_max_masks_nondivisible_k() {
    check_procedural_reduce(StagePlan::new(&[Residence::Smem], StageStorage::Strided, 0));
}

/// `ramp`'s data is nonnegative, so a masked overhang cell falling back to zero (Sum's identity)
/// would still leave `Max` accidentally correct: zero never beats the real maximum. Strictly
/// negative data closes that gap, since a zero fallback would then beat every real value.
#[test]
fn test_reduce_axis_max_nondivisible_k_negative_data() {
    let (m, k, tk) = (4, 6, 4);
    let data: Vec<f32> = (0..m * k).map(|i| -1.0 - i as f32).collect();

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk, Buffering::SINGLE),
        LeafOp::Max,
        &[],
    );

    for i in 0..m {
        let want = data[i * k..(i + 1) * k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got.get_f32(&[i]), want, "Max mismatch at row {i}");
    }
}

/// A zero overhang fallback would incorrectly win this Min reduction, whereas the correct
/// device-side identity is the element type's maximum value.
#[test]
fn test_reduce_axis_min_nondivisible_k_positive_data() {
    let (m, k, tk) = (4, 6, 4);
    let data: Vec<f32> = (0..m * k).map(|i| 1.0 + i as f32).collect();

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk, Buffering::SINGLE),
        LeafOp::Min,
        &[],
    );

    for i in 0..m {
        let want = data[i * k..(i + 1) * k]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert_eq!(got.get_f32(&[i]), want, "Min mismatch at row {i}");
    }
}

#[test]
fn test_reduce_axis_max_outer_axis_retained_innermost_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        space,
        LeafOp::Max,
        4,
        &[],
    );

    for j in 0..k {
        let want: f32 = (0..m)
            .map(|i| ((i * k + j) % 7) as f32)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            got.get_f32(&[j]),
            want,
            "Vectorized max mismatch at index {j}"
        );
    }
}

#[test]
fn test_reduce_axis_min_outer_axis_retained_innermost_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        space,
        LeafOp::Min,
        4,
        &[],
    );

    for j in 0..k {
        let want: f32 = (0..m)
            .map(|i| ((i * k + j) % 7) as f32)
            .fold(f32::INFINITY, f32::min);
        assert_eq!(
            got.get_f32(&[j]),
            want,
            "Vectorized min mismatch at index {j}"
        );
    }
}

/// Reduction over innermost axis with vector_size = 4.
#[test]
fn test_reduce_axis_sum_inner_axis_reduced_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(tm)).axis(K, Cut::sequential(tk))
        })
        .build();

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        space,
        LeafOp::Sum,
        4,
        &[],
    );

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Vectorized sum mismatch at index {i}"
        );
    }
}

/// 3D tensor where the middle axis is reduced and innermost is retained, with vector_size = 4.
#[test]
fn test_reduce_axis_multi_axis_3d_middle_axis_retained_innermost_v4() {
    let (b, m, k) = (3, 4, 16);
    let space = Tiling::new()
        .extents(&[(B, b), (M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(B, Cut::sequential(1))
                .axis(M, Cut::sequential(2))
                .axis(K, Cut::sequential(8))
        })
        .build();

    let got = run_reduce_with_vw(
        shape![b, m, k],
        shape![b, k],
        &[B, M, K],
        &[B, K],
        space,
        LeafOp::Sum,
        4,
        &[],
    );

    for bi in 0..b {
        for ki in 0..k {
            let mut want = 0.0f32;
            for mi in 0..m {
                let flat = bi * (m * k) + mi * k + ki;
                want += (flat % 7) as f32;
            }
            assert_eq!(
                got.get_f32(&[bi, ki]),
                want,
                "3D middle-axis reduce mismatch at ({bi}, {ki})"
            );
        }
    }
}

/// Reduction over an axis spread across plane lanes (ComputeScope::Unit / LaneShare::Plane).
/// Tests that LaneShare::Plane seeds with 0 and folds across lanes combining with accumulator.
#[test]
fn test_reduce_axis_sum_spatial_unit_lanes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;

    let (m, kr) = (4usize, 4usize);
    let k = lanes * kr;
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(m)).axis(K, Cut::unit(kr))
        })
        .build()
        .resolve_lanes(lanes);

    let got = run_reduce(shape![m, k], shape![m], &[M, K], &[M], space, LeafOp::Sum);

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Spatial Unit sum mismatch at index {i}"
        );
    }
}

#[test]
fn test_reduce_axis_max_spatial_unit_lanes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;

    let (m, kr) = (4usize, 4usize);
    let k = lanes * kr;
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(m)).axis(K, Cut::unit(kr))
        })
        .build()
        .resolve_lanes(lanes);

    let got = run_reduce(shape![m, k], shape![m], &[M, K], &[M], space, LeafOp::Max);

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Spatial Unit max mismatch at index {i}"
        );
    }
}

#[test]
fn test_reduce_axis_min_spatial_unit_lanes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let lanes = client.properties().hardware.plane_size_max as usize;

    let (m, kr) = (4usize, 4usize);
    let k = lanes * kr;
    let space = Tiling::new()
        .extents(&[(M, m), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(m)).axis(K, Cut::unit(kr))
        })
        .build()
        .resolve_lanes(lanes);

    let got = run_reduce(shape![m, k], shape![m], &[M, K], &[M], space, LeafOp::Min);

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::INFINITY, f32::min);
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Spatial Unit min mismatch at index {i}"
        );
    }
}
