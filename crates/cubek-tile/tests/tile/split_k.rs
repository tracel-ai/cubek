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

use cubecl::{
    CubeCount, CubeDim, Runtime, TestRuntime,
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};

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
            l.distribute(cubes(CubeAxis::Z), &[(KB, 1)])
                .walk(&[(M, m), (N, n), (KI, inside)])
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
            l.distribute(cubes(CubeAxis::X), &[(M, 1)])
                .walk(&[(N, n), (KB, splits)])
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

/// The precondition the in-kernel combine rests on: this device folds `f32` atomically, and
/// really does fold rather than race, across cubes that all target one cell.
///
/// Probed rather than assumed. `f32` atomic add is there on metal (native and through wgpu's MSL
/// path), CUDA from sm60, and the CPU runtime, but WGSL only has it behind
/// `SHADER_FLOAT32_ATOMIC`, so this is a real fork and not a formality. Nothing else in this file
/// is worth reading if it fails.
#[cube(launch)]
fn atomic_add_probe(out: &mut Tensor<Atomic<f32>>) {
    if UNIT_POS == 0 {
        out[0].fetch_add(1.0);
    }
}

#[test]
fn the_device_folds_floats_atomically_across_cubes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
        return;
    }

    let cubes = 64u32;
    let out = TestInput::builder(client.clone(), shape![1])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    atomic_add_probe::launch::<TestRuntime>(
        &client,
        CubeCount::new_1d(cubes),
        CubeDim::new_single(),
        out.clone().binding().into_tensor_arg(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    // Every cube added one. A lost update reads back low, never high.
    assert_eq!(got.get_f32(&[0]), cubes as f32);
}

// -- The in-kernel combine --------------------------------------------------
//
// The same split, without the second buffer and the second pass: `K` stays one axis, the cubes
// each take a slice of it, and the drain folds each cube's contribution into the output
// atomically. What the workspace pipeline above does in two kernels, this does in one, and the
// two must agree.

const K: Axis = Axis(4);

/// The output is bound as an atomic buffer and drained through a folding sink. `Residence::Register`
/// on it is not decoration: the contraction runs in registers and only the drain touches the
/// destination, which is the one shape a write-only fold admits.
#[cube(launch)]
fn atomic_split_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    out: &AccumulateArg<'_, E>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = out.tile(space);
    let mut acc = c.accumulate::<E, _>(&a, Monoid::Sum);
    acc.mm(&a, &b, Semiring::SUM_PROD);
}

/// `a·b` with `K` dealt out over `splits` cubes, folded atomically into a zeroed output.
fn run_atomic_split_k(m: usize, n: usize, k: usize, splits: usize) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

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
    // Zeroed, and this is where `mm`'s init went: the operation states `c = a·b`, so something
    // has to put the identity there, and under a split that something is the launch.
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.distribute(cubes(CubeAxis::Z), &[(K, k / splits)])
                .walk(&[(M, m), (N, n)])
        })
        .build()
        .with_instruction(Instruction::registers(16));

    atomic_split_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]).residence(&[Residence::Register]),
        ),
        space,
        dtype,
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// The in-kernel combine against the same contraction, unsplit: every cube folds its own slice of
/// `K` into the output and the sum is the whole.
#[test]
fn an_atomic_drain_folds_the_slices_back_together() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);
    let got = run_atomic_split_k(m, n, k, splits);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The two combines agree. Not a restatement of the test above: that one checks the arithmetic,
/// this one checks that the two ways of putting the slices back together are the same program to
/// the caller, which is what makes them interchangeable at launch.
#[test]
fn the_atomic_drain_agrees_with_the_workspace() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
        return;
    }

    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);
    let (_, workspace) = run_split_k(m, n, k, splits);
    let atomic = run_atomic_split_k(m, n, k, splits);
    for i in 0..m {
        for j in 0..n {
            let w = workspace.get_f32(&[i, j]);
            let a = atomic.get_f32(&[i, j]);
            assert!(
                (w - a).abs() < 1e-3,
                "at ({i}, {j}): workspace {w}, atomic {a}"
            );
        }
    }
}

/// The same fold with the lanes carrying cells of their own: `N` rides the plane's lanes while
/// `K` rides the cubes, so a lane owns its columns and a cube owns its slice of the contraction.
///
/// The control on the writer election. A fold from lanes that repeat each other's work has to be
/// made by one of them, and a fold from lanes that each hold their own cells has to be made by
/// all of them: an election that cannot tell the two apart is wrong one way or the other, and
/// this is the half that a blanket "lane zero writes" would silently drop.
#[test]
fn an_atomic_drain_with_lanes_of_their_own() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
        return;
    }
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let dtype = f32::elem_type_native();

    let (m, k, splits, per_lane) = (4usize, 16usize, 4usize, 2usize);
    let n = plane_size * per_lane;

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
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.distribute(lanes(), &[(N, per_lane)])
                .distribute(cubes(CubeAxis::Z), &[(K, k / splits)])
                .walk(&[(M, m)])
        })
        .build()
        .resolve_lanes(plane_size)
        .with_instruction(Instruction::registers(16));

    atomic_split_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]).residence(&[Residence::Register]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The same fold one scope down: `K` cut across the *planes* of a single cube. Planes share no
/// registers either, so each holds a slice of every cell and the drain folds them in, and the
/// election is per plane, so one lane of each folds its own plane's contribution.
///
/// One cube, so nothing here is a cube split at all: what is being checked is that the combine is
/// about instances that cannot meet in registers, not about cubes in particular.
#[test]
fn an_atomic_drain_folds_across_planes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
        return;
    }
    let dtype = f32::elem_type_native();
    let (m, n, k, num_planes) = (4usize, 4usize, 16usize, 4usize);

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
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.distribute(planes(), &[(K, k / num_planes)])
                .walk(&[(M, m), (N, n)])
        })
        .build()
        .with_instruction(Instruction::registers(16));

    atomic_split_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]).residence(&[Residence::Register]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The output contracted *in place*, with no register accumulator at all.
///
/// The verb is still `mm`, and it is still true: across all the cubes the operation is `c = a·b`.
/// What the split moves is the *init* it owns. A cell belongs to several cubes, so none of them
/// may seed it, and the buffer instead arrives holding the fold's identity: zeroed before the
/// launch rather than in the kernel. Every write is then a `+=` into a cell that already holds
/// what the other cubes contracted, and nothing is ever read back, because folding is itself the
/// read-modify-write.
#[cube(launch)]
fn atomic_split_matmul_in_place<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    out: &AccumulateArg<'_, E>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = out.tile(space);
    c.mm(&a, &b, Semiring::SUM_PROD);
}

#[test]
fn a_folding_output_contracts_in_place() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add".to_string(),
        ))
        .enforce();
        return;
    }
    let dtype = f32::elem_type_native();
    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);

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
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.distribute(cubes(CubeAxis::Z), &[(K, k / splits)])
                .walk(&[(M, m), (N, n)])
        })
        .build()
        .with_instruction(Instruction::registers(16));

    atomic_split_matmul_in_place::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            // No residence stated: the contraction happens where the output already lies.
            TileSpec::direct(&[M, N]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}
