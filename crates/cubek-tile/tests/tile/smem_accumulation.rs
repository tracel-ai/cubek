//! A contraction cut across a cube's planes, folded back together in shared memory.
//!
//! `split_k`'s in-kernel combine one scope down: every plane of a cube contracts its own slice of
//! `K` in registers and drains it into the cube's [`SmemAccumulation`], the cube synchronizes, and
//! the sum leaves through a plain store. Nothing reaches global memory but the finished box, so
//! the output is not zeroed first and binds as an ordinary tile.
#![allow(non_snake_case)]

use cubecl::{
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    zspace::shape,
};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};

use super::{Form, implied};
use cubek_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// One box per cube, its `K` distributed to the cube's planes, each plane's slice added into shared
/// memory and the sum stored once.
#[cube(launch)]
fn smem_split_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in &space {
        let mut c_cube = c.at(&cube);
        let sum = c_cube.smem_accumulation::<E>();
        for plane in cube {
            let a_plane = a.at(&plane);
            let b_plane = b.at(&plane);
            let sink_plane = sum.sink.at(&plane);
            let mut partial = sink_plane.block_accumulator::<E, E, E>(
                &a_plane,
                &b_plane,
                REGISTER_BLOCK,
                Monoid::Sum,
            );
            partial.zero();
            partial.mma(&a_plane, &b_plane, Semiring::SUM_PROD);
            partial.drained_into(&sink_plane);
        }
        sync_cube();
        c_cube.copy_from(&sum.source);
    }
}

fn reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| a(i, p) * b(p, j)).sum()
        })
        .collect()
}

/// `a·b` over boxes of `box_m × box_n`, each box's `K` distributed to `planes` planes of its cube.
fn run(m: usize, n: usize, k: usize, (box_m, box_n): (usize, usize), planes: usize) -> HostData {
    let client = cubecl::test_device().client();
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
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, box_m), (N, box_n), (K, k / planes)])
                .planes(&[(K, planes)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::Static,
    );

    smem_split_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[K, N]),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// Whether the device adds `f32` into shared memory atomically; reported rather than silently
/// passed where it cannot.
///
/// The runtime reports the add for buffers, and shared memory is a separate capability it does
/// not report: Vulkan states `shaderSharedFloat32AtomicAdd` apart from the buffer add, and MSL has
/// no threadgroup float atomic at all. So the buffer add is narrowed to the runtimes whose shared
/// memory is known to take it.
fn adds_atomically_in_shared_memory(client: &cubecl::client::Client) -> bool {
    let adds = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
        && matches!(client.name(), "hip" | "cuda" | "cpu");
    if !adds {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no f32 atomic add in shared memory".to_string(),
        ))
        .enforce();
    }
    adds
}

fn assert_matches(got: &HostData, m: usize, n: usize, k: usize) {
    let want = reference(m, n, k);
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

/// One cube, four planes: every cell is the sum of four partials that only ever met in shared
/// memory.
#[test]
fn planes_fold_their_slices_in_shared_memory() {
    if !adds_atomically_in_shared_memory(&cubecl::test_device().client()) {
        return;
    }
    let (m, n, k) = (4, 4, 32);
    assert_matches(&run(m, n, k, (4, 4), 4), m, n, k);
}

/// Several cubes, each with its own accumulator: a box's sum never sees another box's planes.
#[test]
fn each_cube_folds_only_its_own_box() {
    if !adds_atomically_in_shared_memory(&cubecl::test_device().client()) {
        return;
    }
    let (m, n, k) = (8, 12, 24);
    assert_matches(&run(m, n, k, (4, 4), 3), m, n, k);
}

// -- A fragment written into a window the edge cuts short --------------------------------------
//
// A fragment's store intrinsic writes its whole window, so a tile the problem's edge cuts short
// bounces through the plane's scratch and each unit writes only the cells inside.

/// `c = a · b` in `16×16×16` fragments over boxes that overhang the output, drained into a plain
/// output that replaces: through [`Tile::drained_into`], or where `copies` through
/// [`Tile::copy_from`], which bounces the same way.
#[cube(launch)]
fn fragment_matmul_into_a_short_window<EI: Numeric, E: Numeric>(
    a: &TileArg<'_, EI, Const<1>>,
    b: &TileArg<'_, EI, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] copies: bool,
    #[define(EI)] _input: ElemType,
    #[define(E)] _output: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in &space {
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        let mut c_cube = c.at(&cube);
        let mut acc = c_cube
            .cmma_accumulator::<E, EI>(&a_cube, Monoid::Sum)
            .with_scratch(Scratch::OneTile);
        acc.zero();
        let walk = cube.walk();
        let mut stages = Stages::smem(
            &walk,
            &a_cube,
            &b_cube,
            comptime!(StageStorage::Tiled {
                block: vec![(M, 16), (N, 16), (K, 16)],
                chunks: RowChunks::InOrder,
            }),
            1usize,
        );
        stages.pipelined(walk, |slot, stage| {
            let acc_stage = acc.at(stage);
            slot.consume(|a_stage, b_stage| {
                let a_fragments =
                    PlanePartition::<EI>::operand(a_stage, &acc_stage, Instruction::Cmma);
                let b_fragments =
                    PlanePartition::<EI>::operand(b_stage, &acc_stage, Instruction::Cmma);
                for fragment in stage.walk().unrolled() {
                    let mut acc_fragment = acc_stage.at(&fragment);
                    acc_fragment.mma(
                        &a_fragments.at(&fragment),
                        &b_fragments.at(&fragment),
                        Semiring::SUM_PROD,
                    );
                }
            });
        });
        if comptime!(copies) {
            c_cube.copy_from(&acc);
        } else {
            acc.drained_into(&c_cube);
        }
    }
}

/// Whether this device contracts `16×16×16` `f16` fragments summed in `f32`; reported rather than
/// silently passed where it cannot.
fn contracts_f16_fragments(client: &cubecl::client::Client) -> bool {
    let f16 = half::f16::elem_type_native();
    let f32 = f32::elem_type_native();
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == f16
            && cfg.b_type == f16
            && cfg.cd_type == f32
            && (cfg.m, cfg.n, cfg.k) == (16, 16, 16)
    });
    if !offered {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no 16x16x16 f16 cmma fragment summed in f32".to_string(),
        ))
        .enforce();
    }
    offered
}

/// A `20 × 24` product in `16 × 16` boxes: every box but one overhangs the output, and the cells
/// past the edge are never written, so the output reads the product and nothing else.
#[test]
fn a_fragment_drains_into_a_window_the_edge_cuts_short() {
    short_window_matmul(false);
}

/// The same product stored with `copy_from`, which bounces a fragment into a short window just as
/// the drain does.
#[test]
fn a_fragment_copies_into_a_window_the_edge_cuts_short() {
    short_window_matmul(true);
}

fn short_window_matmul(copies: bool) {
    let client = cubecl::test_device().client();
    if !contracts_f16_fragments(&client) {
        return;
    }
    let (m, n, k) = (20usize, 24usize, 32usize);
    let input = half::f16::elem_type_native();
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(input)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(input)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 16), (N, 16), (K, 16)])
                .walk(&[(M, 1), (N, 1)])
                .walk_every(&[K])
                .cubes(&[M, N])
                .build(),
        ),
        Form::Static,
    );
    // Bound through the launcher, which reads the overhang off the partitioning and masks the
    // output's writes past its edge.
    let bind = |binding, axes: &'static [Axis]| launcher.arg(binding).axes(axes).build();
    fragment_matmul_into_a_short_window::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        bind(a_handle.clone().binding(), &[M, K]).arg(),
        bind(b_handle.clone().binding(), &[K, N]).arg(),
        bind(out.clone().binding(), &[M, N]).arg(),
        launcher.partitioning_arg(),
        copies,
        input,
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    assert_matches(&got, m, n, k);
}
