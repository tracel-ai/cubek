//! `c.mm(&a.scaled(&ComptimeOption::new_Some(s)), &b, semiring)`: the contraction with one
//! factor scaled by a **real operand**, on the factor the kernel wrote it on.
//!
//! *Which* operand is not stated: the scales' own axes say it. A scale over the output's columns
//! is a fact about the rhs's columns and nothing else could fold it in; anything else scales the
//! lhs. One verb serves both: `(a ⊗ s) · b` and `a · (b ⊗ s)` are one sum, folded where cheapest.
//!
//! The point is what the kernel signature says: the values are a tensor of values, the scales a
//! tensor of scales, both named at the call, and the fold is a verb the kernel writes. Nothing
//! decodes behind a read or rides a side channel, so `f16` scales work with no widening pass.
//!
//! The scales resolve at their own granularity through their own projection: a plain `KB` for a
//! per-block scale, `KI` too for a per-element one, an omitted axis for a broadcast.

use cubecl::{
    bytes::Bytes,
    prelude::*,
    quant::scheme::{QuantValue, ScaleDtype},
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubecl_common::{e2m1, e4m3};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;
use half::f16;

use super::matmul::require_cmma_8x8x8_f32;

/// Which factor a test kernel writes its scales on. The engine has no such enum: a kernel says
/// which by where it writes `.scaled()`, and these kernels serve both cases from one launch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Scaled {
    Lhs,
    Rhs,
}

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// The contraction, as the two axes a block makes of it: which block, and where inside it.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// The register block the contractions here run under, but for the wide-lhs one.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `c = (a ⊗ s) · b`, with `s` at whatever granularity its own projection states.
#[cube(launch)]
fn scaled_matmul<E: Numeric, S: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, S, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scale = scale.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        match comptime!(side) {
            Scaled::Lhs => c_r.mma_scaled_with(
                &a.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                &b.at(&region).plain(),
                REGISTER_BLOCK,
                Semiring::SUM_PROD,
            ),
            Scaled::Rhs => c_r.mma_scaled_with(
                &a.at(&region).plain(),
                &b.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                REGISTER_BLOCK,
                Semiring::SUM_PROD,
            ),
        }
    }
}

/// [`scaled_matmul`] with the accumulator promoted to registers: the partials never round-trip
/// through `c`'s element between `K` steps, which is the form a decode gemv wants.
#[cube(launch)]
fn scaled_matmul_promoted<E: Numeric, S: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, S, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scale = scale.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.block_accumulator::<E, E, E>(
        &a,
        &b,
        comptime!(Fragments::new(
            &c.space,
            &a.space,
            std::slice::from_ref(&level)
        )),
        REGISTER_BLOCK,
        Monoid::Sum,
    );
    acc.zero();
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        match comptime!(side) {
            Scaled::Lhs => acc_r.mma_scaled(
                &a.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                &b.at(&region).plain(),
                Semiring::SUM_PROD,
            ),
            Scaled::Rhs => acc_r.mma_scaled(
                &a.at(&region).plain(),
                &b.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                Semiring::SUM_PROD,
            ),
        }
    }
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// [`scaled_matmul`] with two scale levels: block scales, and one factor over the whole tensor.
#[cube(launch)]
fn two_level_scaled_matmul<E: Numeric, S: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    blocks: &TileArg<'_, S, Const<1>>,
    global: &TileArg<'_, S, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    // The operand is the hierarchy: block scales, under the factor that covers a tile of their
    // tiles. Nothing states a scheme.
    let blocks = blocks.tile(comptime!(space.clone()));
    let global = global.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        match comptime!(side) {
            Scaled::Lhs => c_r.mma_scaled_with(
                &a.at(&region)
                    .scaled(&ComptimeOption::new_Some(blocks.at(&region)))
                    .scaled(&ComptimeOption::new_Some(global.at(&region))),
                &b.at(&region).plain(),
                REGISTER_BLOCK,
                Semiring::SUM_PROD,
            ),
            Scaled::Rhs => c_r.mma_scaled_with(
                &a.at(&region).plain(),
                &b.at(&region)
                    .scaled(&ComptimeOption::new_Some(blocks.at(&region)))
                    .scaled(&ComptimeOption::new_Some(global.at(&region))),
                REGISTER_BLOCK,
                Semiring::SUM_PROD,
            ),
        }
    }
}

/// [`scaled_matmul`] on a tensor-core accumulator: the scaled operand is landed in shared memory
/// by the plane's lanes, unpacked and scaled, and loaded as the fragment the plain instruction
/// takes. Both operands carry a landing here so one kernel serves either side.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn scaled_matmul_cmma<E: Numeric, S: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, S, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone())).with_landing();
    let b = b.tile(comptime!(space.clone())).with_landing();
    let scale = scale.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.cmma_accumulator::<E, E>(
        &a,
        comptime!(Fragments::new(
            &c.space,
            &a.space,
            std::slice::from_ref(&level)
        )),
        Monoid::Sum,
    );
    acc.zero();
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        match comptime!(side) {
            Scaled::Lhs => acc_r.mma_scaled(
                &a.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                &b.at(&region).plain(),
                Semiring::SUM_PROD,
            ),
            Scaled::Rhs => acc_r.mma_scaled(
                &a.at(&region).plain(),
                &b.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                Semiring::SUM_PROD,
            ),
        }
    }
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// **Two scale levels, applied in order.** `nvfp4`'s shape: a scale per block of the contraction,
/// under one factor for the whole tensor.
///
/// The second level is an operand like the first, spanning the same axes and distinguishing
/// fewer of them, and it rides the same [`Scales`] as the block level. Nothing in the kernel
/// counts levels: it windows `scales` and hands it to the leaf.
#[test]
fn two_levels_fold_in_order() {
    let (rows, cols, block, blocks) = (4, 4, 8, 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let g = vec![0.25f32];

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let (g_t, _) = TestInput::builder(client.clone(), shape![1])
        .dtype(dtype)
        .custom(g.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    two_level_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            g_t.binding().into_tensor_arg(),
            // Same axes as the level below it, addressing neither: one value over all their tiles.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::broadcast()])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * g[0] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// One block per region, so each region carries one scale per row.
#[test]
fn a_scaled_contraction_folds_the_block_scale_in() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    // Small integers, so the reference is exact.
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    // The values' projection names the block; the scales' is derived from it, one per `KB`.
    let a_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(a_projection.clone()),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(a_projection.scales_per(KB)),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// A region inside one block: several of them share a scale, because they share a `KB`.
#[test]
fn a_cut_finer_than_the_block_reuses_its_scale() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block / 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **A scale over no axis at all.** The base of the hierarchy: one number covering everything,
/// which is what a per-tensor level is.
///
/// Spelled the way every other granularity is, by which axes the operand distinguishes. The nest
/// still names `[M, KB]`, so the scales' matrix has the shape any other level's would; the
/// projection addresses neither, so every position reads the same value. Nothing says "per tensor".
#[test]
fn a_scale_over_no_axis_covers_everything() {
    let (rows, cols, block, blocks) = (4, 4, 8, 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s = vec![0.5f32];

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![1])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            // One physical axis, addressing no logical one: every position resolves to index 0.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::broadcast()])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[0] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// A region spanning two blocks: the scale changes within it, and the step's own `KB` coordinate
/// is what picks which.
#[test]
fn a_cut_coarser_than_the_block_changes_scale_within_a_region() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (2, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The one that pays for the design.** The scales are an `f16` tensor and the kernel reads them
/// as one: no widening pass, no scheme saying otherwise, no way for the two to disagree. A scale
/// is whatever its tensor holds because it is a tensor. The values stay `f32`, `s` exact in both.
#[test]
fn f16_scales_are_read_as_f16() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let scale_dtype = f16::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(scale_dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, scale_dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The scales span `N`, so they scale the rhs, and the call site says so
/// on the rhs; the leaf checks that against the axes.
#[test]
fn scales_over_the_columns_scale_the_rhs() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            // Spanning `N` is what makes it the rhs's.
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Rhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The rhs scale under a cut finer than its block: several regions share it.
#[test]
fn an_rhs_scale_survives_a_finer_cut() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block / 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Rhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// And under a coarser one, where the scale changes within a region.
#[test]
fn an_rhs_scale_changes_within_a_coarser_region() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (2, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Rhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The gap the decode gemv was waiting on.** The kernel opens a register block above the whole
/// walk, so the scaled steps fold into it directly: refused before, and the same numbers as the
/// memory-backed form.
#[test]
fn a_promoted_accumulator_takes_the_scaled_contraction() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    scaled_matmul_promoted::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`scaled_matmul_promoted`] with the rhs's scales served as lines: `SW` of them per read,
/// along `N`.
#[cube(launch)]
fn wide_rhs_scaled_matmul_promoted<E: Numeric, S: Numeric, SW: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, S, SW>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scale = scale.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.block_accumulator::<E, E, E>(
        &a,
        &b,
        comptime!(Fragments::new(
            &c.space,
            &a.space,
            std::slice::from_ref(&level)
        )),
        REGISTER_BLOCK,
        Monoid::Sum,
    );
    acc.zero();
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        match comptime!(side) {
            Scaled::Lhs => acc_r.mma_scaled(
                &a.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                &b.at(&region).plain(),
                Semiring::SUM_PROD,
            ),
            Scaled::Rhs => acc_r.mma_scaled(
                &a.at(&region).plain(),
                &b.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                Semiring::SUM_PROD,
            ),
        }
    }
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// **Scales served as lines along the columns, into a promoted accumulator.** The twin of
/// [`lhs_scales_are_served_several_at_a_time`].
///
/// The case the two old asserts disagreed about: the memory nest admitted a wide scale line where
/// its step folded one contracted value, the promoted block where the scales rode the rhs, and
/// nothing exercised the second.
///
/// The block walks its columns under a constant ordinal, which is what a wide read needs — a fold
/// is a lane of the read it arrived in, and a lane index is not addressable at runtime. So lane
/// `j` of a scale line goes with column `j`, and the scales vary per `(block of K, column)`.
#[test]
fn rhs_scales_are_served_several_at_a_time() {
    let (rows, cols, block, blocks, lanes) = (2, 4, 8, 4, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Distinct per (block of `K`, column), halves so the reference is exact.
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    wide_rhs_scaled_matmul_promoted::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        lanes,
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            // `N` innermost: the width lands on the axis the scales vary over fastest, and
            // spanning it is what makes them the rhs's.
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Rhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`scaled_matmul`] with the lhs's scales served as lines: `SW` of them per read, along `KB`.
/// The fold rides the lane walk, which reads one line per `lw` steps and takes a fixed component
/// of it; the scalar walk has no line ordinal to fold under, so this block fans out over lanes.
#[cube(launch)]
fn wide_lhs_scaled_matmul<E: Numeric, S: Numeric, SW: Size>(
    a: &TileArg<'_, E, Const<4>>,
    b: &TileArg<'_, E, Const<1>>,
    scale: &TileArg<'_, S, SW>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scale = scale.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        match comptime!(side) {
            Scaled::Lhs => c_r.mma_scaled_with(
                &a.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                &b.at(&region).plain(),
                comptime!(RegisterBlock::new(64).lane_fanout()),
                Semiring::SUM_PROD,
            ),
            Scaled::Rhs => c_r.mma_scaled_with(
                &a.at(&region).plain(),
                &b.at(&region)
                    .scaled(&ComptimeOption::new_Some(scale.at(&region))),
                comptime!(RegisterBlock::new(64).lane_fanout()),
                Semiring::SUM_PROD,
            ),
        }
    }
}

/// **Scales served as lines along the contraction.** The lhs's scales vary over `KB` and nothing
/// else, so `KB` is their innermost axis and a read serves several blocks of `K` at once.
///
/// The block walks its contraction in runs of one scale line for this: the folds are unrolled so
/// each one's lane is a constant, and the lines under one fold stay rolled, since they all take
/// the same scale. One row here, so the scales are per block of `K` alone.
#[test]
fn lhs_scales_are_served_several_at_a_time() {
    let (cols, block, blocks, lanes) = (4, 4, 4, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Distinct per block of `K`, halves so the reference is exact.
    let s: Vec<f32> = (0..blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, 1), (N, cols), (KB, blocks), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );

    wide_lhs_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        lanes,
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            // `KB` innermost and alone: the width lands on the axis the scales vary over.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::of(KB)])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        Scaled::Lhs,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth)
            .map(|k| a[k] * s[k / block] * b[k * cols + n])
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at {n}: got {have}, want {want}"
        );
    }
}

/// Which operand of the tensor-core test carries the scales, and how the rhs is stored.
#[derive(Clone, Copy)]
enum CmmaCase {
    /// Scales per `(row, block of K)`, on the lhs.
    Lhs,
    /// Scales per `(block of K, column)`, on a rhs stored `{K, N}`.
    RhsRowMajor,
    /// The same scales on a rhs stored `{N, K}`, read col-major, so the landing is the
    /// transpose of the scales' matrix.
    RhsColMajor,
}

/// `c = (a ⊗ s) · b` or `a · (b ⊗ s)` through the M2's `8x8x8` fragment, one block of `K` a
/// region, against the host product.
fn check_scaled_cmma(case: CmmaCase) {
    let (rows, cols, block, blocks) = (8, 8, 8, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * 8).map(|i| (i as f32 + 1.0) / 2.0).collect();
    // The rhs as stored: `{K, N}`, or `{N, K}` holding the same matrix.
    let b_stored: Vec<f32> = match case {
        CmmaCase::RhsColMajor => (0..cols * depth)
            .map(|i| b[(i % depth) * cols + i / depth])
            .collect(),
        _ => b.clone(),
    };
    let b_shape = match case {
        CmmaCase::RhsColMajor => shape![cols, depth],
        _ => shape![depth, cols],
    };
    let s_shape = match case {
        CmmaCase::Lhs => shape![rows, blocks],
        _ => shape![blocks, cols],
    };

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), b_shape)
        .dtype(dtype)
        .custom(b_stored)
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), s_shape)
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );
    let split = PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]);
    let b_spec = match case {
        CmmaCase::RhsColMajor => TileSpec::new(Projection::new(
            &[N, KB, KI],
            &[PhysicalAxisMap::of(N), split.clone()],
        )),
        _ => TileSpec::new(Projection::new(
            &[KB, KI, N],
            &[split.clone(), PhysicalAxisMap::of(N)],
        )),
    };
    let s_spec = match case {
        CmmaCase::Lhs => TileSpec::new(Projection::new(
            &[M, KB],
            &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
        )),
        _ => TileSpec::new(Projection::new(
            &[KB, KI, N],
            &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
        )),
    };
    let side = match case {
        CmmaCase::Lhs => Scaled::Lhs,
        _ => Scaled::Rhs,
    };

    scaled_matmul_cmma::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), split],
            )),
        ),
        TileArgLaunch::new(b_t.binding().into_tensor_arg(), b_spec),
        TileArgLaunch::new(s_t.binding().into_tensor_arg(), s_spec),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        side,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    let scale = match case {
                        CmmaCase::Lhs => s[m * blocks + k / block],
                        _ => s[(k / block) * cols + n],
                    };
                    a[m * depth + k] * scale * b[k * cols + n]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The scaled contraction runs on the tensor cores.** The lhs is landed scaled by the plane's
/// lanes and loaded as the `A` fragment; the instruction is the plain one.
#[test]
fn a_cmma_accumulator_takes_the_scaled_contraction() {
    check_scaled_cmma(CmmaCase::Lhs);
}

/// The rhs side of the same, stored `{K, N}`: the landing is the `B` fragment as stored.
#[test]
fn a_cmma_accumulator_takes_rhs_scales() {
    check_scaled_cmma(CmmaCase::RhsRowMajor);
}

/// The rhs stored `{N, K}`, the way a weight lies with its contraction innermost: the landing is
/// col-major, so each line's scale sits at its column and its row's block.
#[test]
fn a_cmma_accumulator_takes_rhs_scales_col_major() {
    check_scaled_cmma(CmmaCase::RhsColMajor);
}

/// [`scaled_matmul_cmma`] with the rhs **staged as its words** before it lands: the packed
/// window is copied into shared memory verbatim, the stage keeps the packing, and the landing
/// unpacks and scales out of the stage exactly as it would out of the global window.
#[cube(launch)]
fn scaled_matmul_cmma_staged<E: Numeric, S: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, u32, Const<1>>,
    scale: &TileArg<'_, S, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone())).with_landing();
    let b = b.tile_as::<E>(comptime!(space.clone()));
    let scale = scale.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut stage = MemData::<E>::stage(
        &b,
        comptime!(level.clone()),
        StageStorage::Strided,
        comptime!(None),
    )
    .with_landing();
    let mut acc = c.cmma_accumulator::<E, E>(
        &a,
        comptime!(Fragments::new(
            &c.space,
            &a.space,
            std::slice::from_ref(&level)
        )),
        Monoid::Sum,
    );
    acc.zero();
    for region in space.over(&level) {
        stage.copy_from(&b.at(&region));
        sync_cube();
        let mut acc_r = acc.at(&region);
        acc_r.mma_scaled(
            &a.at(&region).plain(),
            &stage.scaled(&ComptimeOption::new_Some(scale.at(&region))),
            Semiring::SUM_PROD,
        );
        // The stage is refilled next region, once every plane has landed from it.
        sync_cube();
    }
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// **A packed stage lands on the tensor cores.** `e2m1` values eight to a word, stored `{N, K}`
/// with the contraction innermost as a weight lies, staged one scale block at a time as the words
/// they are, then unpacked and scaled into the fragment's landing: a quarter the stage, one answer.
#[test]
fn a_packed_stage_lands_on_the_tensor_cores() {
    let (rows, cols, block, blocks) = (8, 8, 8, 4);
    let depth = block * blocks;
    let field = QuantValue::E2M1;
    let factor = 32 / field.size_bits();

    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    // The rhs as stored, `{N, K}`: every `e2m1` code, cycled, eight to a word down `k`.
    let codes: Vec<u32> = (0..cols * depth).map(|i| (i % 16) as u32).collect();
    let words: Vec<u32> = codes
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &c)| acc | (c << (j * field.size_bits())))
        })
        .collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let b_t = TensorHandle::new_contiguous(
        vec![cols, depth / factor],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = Launcher::implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Tiling::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .levels(),
        ),
        KernelForm::Static,
    );
    let split = PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]);

    scaled_matmul_cmma_staged::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), split.clone()],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[N, KB, KI],
                &[PhysicalAxisMap::of(N), split],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.level(0),
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    let b = e2m1::from_bits(codes[n * depth + k] as u8).to_f32();
                    a[m * depth + k] * s[(k / block) * cols + n] * b
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The columns, as the two axes a storage tile makes of them: which tile, and where inside it.
const NB: Axis = Axis(4);
const NI: Axis = Axis(5);

/// `c = a · (b ⊗ s)` over a weight **stored in tile order**, walked the way the memory-bound
/// kernel walks it on either arm: the cube grid, the planes, the chunks a plane walks under one
/// load of its scales, the steps of a chunk, and below them the block's lanes or the one fragment.
///
/// The scales of a chunk are loaded once, into the plane's lanes or its own shared window, and
/// every step reads its scale from there.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn chunked_scaled_matmul<E: Numeric, S: Numeric, SS: Numeric>(
    a: &TileArg<'_, E, Const<8>>,
    b: &TileArg<'_, u32, Const<1>>,
    scale: &TileArg<'_, SS, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] instruction: Instruction,
    #[comptime] chunks: Level,
    #[comptime] reach: Reach,
    #[comptime] cells: Option<Level>,
    #[define(E, S, SS)] _dtypes: [ElemType; 3],
) {
    // Both factors land where the instruction reads a window as it lies, and neither does where
    // it reads through a layout; `landed_for` is where that rule lives.
    let a = a.tile(comptime!(space.clone())).landed_for(instruction);
    let b = b
        .tile_as::<E>(comptime!(space.clone()))
        .landed_for(instruction);
    let scale = scale.tile_as::<S>(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        let scale_cube = scale.at(&cube);
        let c_cube = c.at(&cube);
        for plane in cube {
            let a_plane = a_cube.at(&plane);
            let b_plane = b_cube.at(&plane);
            let scale_plane = scale_cube.at(&plane);
            let c_plane = c_cube.at(&plane);
            let mut lines = MemData::<S>::stage(
                &scale_plane,
                comptime!(chunks.clone()),
                comptime!(StageStorage::Lanes { reach }),
                comptime!(None),
            );
            let mut sum = c_plane.accumulator::<E, E, E>(
                &a_plane,
                &b_plane,
                comptime!(Fragments::below(&c_plane, &a_plane)),
                instruction,
                Monoid::Sum,
            );
            sum.zero();
            for chunk in plane {
                lines.copy_from(&scale_plane.at(&chunk));
                for step in chunk {
                    for leaf in step {
                        let mut sum_leaf = sum.at(&leaf);
                        sum_leaf.mma_scaled(
                            &a_plane.at(&leaf).plain(),
                            &b_plane.at(&leaf).scaled_by(lines.at(&leaf)),
                            Semiring::SUM_PROD,
                        );
                    }
                }
            }
            sum.drained_into(&c_plane, comptime!(cells.clone()));
        }
    }
}

/// Row-major strides over `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// The scales of a tile-ordered weight as a test stores them: which float, and the words.
#[derive(Clone, Copy, Debug)]
enum TileScales {
    /// Whole `f32`s, one a word.
    F32,
    /// `ue4m3` bytes, four to a word: what NVFP4 carries.
    Ue4m3,
}

/// A weight stored in tile order, with its activation and the host's view of both.
///
/// The values lie `[NB][KB][NI][KI]`: a tile is sixteen columns of sixteen `k`, a column eight
/// bytes. The scales lie `[NB][KB][NI]`: a tile's sixteen scales are one line, one per column,
/// and the tiles down `k` follow one another, so a chunk of them is consecutive lines.
struct TileOrdered {
    rows: usize,
    tile: usize,
    n_tiles: usize,
    k_tiles: usize,
    a: Vec<f32>,
    words: Vec<u32>,
    s: Vec<f32>,
}

impl TileOrdered {
    const FIELD: QuantValue = QuantValue::E2M1;

    fn new(rows: usize, n_tiles: usize, k_tiles: usize) -> Self {
        let tile = 16;
        let depth = k_tiles * tile;
        let factor = 32 / Self::FIELD.size_bits();
        let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
        let this = TileOrdered {
            rows,
            tile,
            n_tiles,
            k_tiles,
            a,
            words: Vec::new(),
            s: Vec::new(),
        };
        // The words as stored: tile order, a column's sixteen codes eight to a word, low nibble
        // first.
        let mut words = Vec::with_capacity(n_tiles * k_tiles * tile * tile / factor);
        for nb in 0..n_tiles {
            for kb in 0..k_tiles {
                for ni in 0..tile {
                    for word in 0..tile / factor {
                        words.push((0..factor).fold(0u32, |acc, j| {
                            acc | ((this.code(nb, kb, ni, word * factor + j) as u32)
                                << (j * Self::FIELD.size_bits()))
                        }));
                    }
                }
            }
        }
        // One scale per column per tile: halves from 0.5 to 4, exact in `e4m3`, and different
        // between neighbouring columns *and* neighbouring tiles, so a scale read at the wrong
        // column or the wrong block is a wrong number rather than the same one.
        let s: Vec<f32> = (0..n_tiles * k_tiles * tile)
            .map(|i| ((i % 8) + (i / 8) + (i / 64)) as f32 % 8.0 / 2.0 + 0.5)
            .collect();
        TileOrdered { words, s, ..this }
    }

    fn depth(&self) -> usize {
        self.k_tiles * self.tile
    }

    fn cols(&self) -> usize {
        self.n_tiles * self.tile
    }

    /// The `e2m1` code at `(nb, kb, ni, ki)`: every code, and a different run of them in every
    /// column of every tile, so a value read from the wrong column or the wrong tile is a wrong
    /// number rather than the same one.
    fn code(&self, nb: usize, kb: usize, ni: usize, ki: usize) -> usize {
        (ki + 3 * ni + 5 * kb + 7 * nb) % 16
    }

    /// The scale at `(nb, kb, ni)`, as stored.
    fn scale_at(&self, nb: usize, kb: usize, ni: usize) -> f32 {
        self.s[(nb * self.k_tiles + kb) * self.tile + ni]
    }

    /// The product on the host.
    fn want(&self, m: usize, n: usize) -> f32 {
        let (nb, ni) = (n / self.tile, n % self.tile);
        (0..self.depth())
            .map(|k| {
                let (kb, ki) = (k / self.tile, k % self.tile);
                let b = e2m1::from_bits(self.code(nb, kb, ni, ki) as u8).to_f32();
                self.a[m * self.depth() + k] * b * self.scale_at(nb, kb, ni)
            })
            .sum()
    }

    fn space(&self) -> Space {
        Space::new(&[
            (M, self.rows),
            (NB, self.n_tiles),
            (NI, self.tile),
            (KB, self.k_tiles),
            (KI, self.tile),
        ])
    }

    /// The activation, served at the width a word of the weight unpacks to.
    fn a_op(&self, client: &Client, launcher: &Launcher) -> StridedOperand {
        let (a_t, _) = TestInput::builder(client.clone(), shape![self.rows, self.depth()])
            .dtype(f32::elem_type_native())
            .custom(self.a.clone())
            .generate_with_f32_host_data();
        launcher
            .arg(a_t.binding())
            .gathered(
                Projection::dims()
                    .dim(M)
                    .dim(split(&[(KB, self.k_tiles), (KI, self.tile)]))
                    .build(),
            )
            .vectorize(32 / Self::FIELD.size_bits())
            .build()
    }

    /// The weight as stored. A packed binding counts values, its words being the packing's
    /// business: the shape and the strides are the tiles' in values.
    fn b_op(&self, client: &Client, launcher: &Launcher) -> StridedOperand {
        let shape = vec![self.n_tiles, self.k_tiles, self.tile, self.tile];
        let b_t = TensorHandle::new_contiguous(
            shape.clone(),
            client.create(Bytes::from_elems(self.words.clone())),
            u32::elem_type_native(),
        );
        let mut binding = b_t.binding();
        binding.shape = shape.clone().into();
        binding.strides = contiguous_strides(&shape).into();
        launcher
            .arg(binding)
            .gathered(Projection::dims().dim(NB).dim(KB).dim(NI).dim(KI).build())
            .packed(Self::FIELD)
            .build()
    }

    /// The scales as stored, served as `f32` a line (a tile's sixteen) a read: whole words, or
    /// `ue4m3` bytes four to a word, the same shape in values either way. Returns the element
    /// they are stored as.
    fn s_op(
        &self,
        client: &Client,
        launcher: &Launcher,
        scales: TileScales,
    ) -> (StridedOperand, ElemType) {
        let axes = Projection::dims()
            .dim(NB)
            .dim(KB)
            .dim(NI)
            .spanning(KI)
            .build();
        let shape = vec![self.n_tiles, self.k_tiles, self.tile];
        match scales {
            TileScales::F32 => {
                let (s_t, _) = TestInput::builder(
                    client.clone(),
                    shape![self.n_tiles, self.k_tiles, self.tile],
                )
                .dtype(f32::elem_type_native())
                .custom(self.s.clone())
                .generate_with_f32_host_data();
                (
                    launcher
                        .arg(s_t.binding())
                        .gathered(axes)
                        .vectorize(self.tile)
                        .build(),
                    f32::elem_type_native(),
                )
            }
            TileScales::Ue4m3 => {
                let per_word = 4;
                let bytes: Vec<u32> = self
                    .s
                    .chunks(per_word)
                    .map(|word| {
                        word.iter().enumerate().fold(0u32, |acc, (j, &v)| {
                            acc | ((e4m3::from_f32(v).to_bits() as u32) << (j * 8))
                        })
                    })
                    .collect();
                let s_t = TensorHandle::new_contiguous(
                    shape.clone(),
                    client.create(Bytes::from_elems(bytes)),
                    u32::elem_type_native(),
                );
                let mut binding = s_t.binding();
                binding.shape = shape.clone().into();
                binding.strides = contiguous_strides(&shape).into();
                (
                    launcher
                        .arg(binding)
                        .gathered(axes)
                        .packed(scale_field(ScaleDtype::UE4M3))
                        .vectorize(self.tile)
                        .build(),
                    u32::elem_type_native(),
                )
            }
        }
    }

    fn c_op(&self, launcher: &Launcher, c: &TensorHandle) -> StridedOperand {
        launcher
            .arg(c.clone().binding())
            .gathered(
                Projection::dims()
                    .dim(M)
                    .dim(split(&[(NB, self.n_tiles), (NI, self.tile)]))
                    .build(),
            )
            .build()
    }

    fn check(&self, client: &Client, c: TensorHandle, what: &str) {
        let got = HostData::from_tensor_handle(client, c, HostDataType::F32);
        for m in 0..self.rows {
            for n in 0..self.cols() {
                let want = self.want(m, n);
                let have = got.get_f32(&[m, n]);
                assert!(
                    (have - want).abs() < 1e-2 * want.abs().max(1.0),
                    "{what} at ({m}, {n}): got {have}, want {want}"
                );
            }
        }
    }
}

/// Which arm a chunked contraction runs on.
#[derive(Clone, Copy, Debug)]
enum Arm {
    Registers,
    Landing,
}

/// **A plane holds its scales for a chunk.** The weight lies in tile order and its scales in
/// lines, `[NB][KB][NI]`; a plane walks the contraction a chunk of thirty-two blocks at a time and
/// loads the chunk's thirty-two lines once: lane `t` holds line `t`, or a shared window holds all.
///
/// Every step reads the scale of the value it lands or contracts at that value's coordinates, a
/// word at a time.
///
/// On the register arm a lane holds one column over one block a step, the plane's lanes are a
/// tile's columns by two blocks, and a chunk is sixteen steps. On the tensor cores a plane holds
/// one fragment, eight rows by half a tile's columns, and walks a chunk one fragment depth a step.
fn check_chunked(arm: Arm, scales: TileScales, reach: Reach) {
    let (rows, n_tiles, chunk, chunks) = (8, 2, 32, 2);
    let w = TileOrdered::new(rows, n_tiles, chunk * chunks);
    let client = cubecl::test_device().client();
    if matches!(arm, Arm::Landing) && !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let c = TestInput::builder(client.clone(), shape![rows, w.cols()])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // Leaf up: what a lane or a plane holds a step, the steps of a chunk, the chunks, the planes
    // of a cube, the cubes. A fragment is a level of one all the same, since the body's innermost
    // loop is a level's.
    let levels = match arm {
        Arm::Registers => Tiling::leaf(&[(M, rows), (NI, 1), (KB, 1), (KI, w.tile)])
            .lanes(&[(NI, w.tile), (KB, 2)])
            .walk(&[(KB, chunk / 2)])
            .walk_every(&[KB])
            .planes(&[(NB, 1)])
            .cubes(&[NB])
            .levels(),
        Arm::Landing => Tiling::leaf(&[(M, rows), (NI, 8), (KI, 8)])
            .walk(&[(NI, 1)])
            .walk(&[(KB, chunk), (KI, 2)])
            .walk_every(&[KB])
            .planes(&[(NI, w.tile / 8)])
            .cubes(&[NB])
            .levels(),
    };
    let chunks_level = levels[2].clone();
    let cells = match arm {
        Arm::Registers => Some(levels[4].clone()),
        Arm::Landing => None,
    };
    let instruction = match arm {
        Arm::Registers => Instruction::Registers {
            config: REGISTER_BLOCK,
        },
        Arm::Landing => Instruction::Cmma,
    };
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(w.space(), levels),
        KernelForm::Static,
    );
    let (s_op, stored) = w.s_op(&client, &launcher, scales);

    chunked_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w.a_op(&client, &launcher).arg(),
        w.b_op(&client, &launcher).arg(),
        s_op.arg(),
        w.c_op(&launcher, &c).arg(),
        launcher.partitioning_arg(),
        instruction,
        chunks_level,
        reach,
        cells,
        [dtype, dtype, stored],
    );
    w.check(&client, c, &format!("{arm:?} {scales:?}"));
}

/// Every arm and every scale element, **at both reaches**: what the lanes hold is the same
/// either way, so a value read by shuffle and a value read out of the plane's window are the
/// same value or one of the two is wrong.
#[test]
fn a_plane_holds_its_scales_in_its_lanes() {
    for reach in [Reach::Shuffle, Reach::Window] {
        check_chunked(Arm::Registers, TileScales::F32, reach);
    }
}

#[test]
fn a_plane_holds_its_byte_scales_in_its_lanes() {
    for reach in [Reach::Shuffle, Reach::Window] {
        check_chunked(Arm::Registers, TileScales::Ue4m3, reach);
    }
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores() {
    for reach in [Reach::Shuffle, Reach::Window] {
        check_chunked(Arm::Landing, TileScales::F32, reach);
    }
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores_under_byte_scales() {
    for reach in [Reach::Shuffle, Reach::Window] {
        check_chunked(Arm::Landing, TileScales::Ue4m3, reach);
    }
}

/// `c = a · (b ⊗ s)` over a weight stored in tile order, walked as the compute-bound body walks
/// it on the tensor cores: a plane holds a partition of fragments (two rows of two columns) and at
/// every step lands each factor's window once, the weight unpacked and scaled by the chunk's lines.
///
/// The partition's fragments then load from the landing a depth at a time, the quant block being
/// the loop inside the partition's depth: two instructions under one scale.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn partitioned_scaled_matmul<E: Numeric, S: Numeric, SS: Numeric>(
    a: &TileArg<'_, E, Const<8>>,
    b: &TileArg<'_, u32, Const<1>>,
    scale: &TileArg<'_, SS, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] chunks: Level,
    #[comptime] reach: Reach,
    #[comptime] grid: Level,
    #[define(E, S, SS)] _dtypes: [ElemType; 3],
) {
    let a = a.tile(comptime!(space.clone())).with_landing();
    let b = b.tile_as::<E>(comptime!(space.clone())).with_landing();
    let scale = scale.tile_as::<S>(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        let scale_cube = scale.at(&cube);
        let c_cube = c.at(&cube);
        for plane in cube {
            let a_plane = a_cube.at(&plane);
            let b_plane = b_cube.at(&plane);
            let scale_plane = scale_cube.at(&plane);
            let c_plane = c_cube.at(&plane);
            let out = comptime!(c_plane.space.clone());
            let mut lines = MemData::<S>::stage(
                &scale_plane,
                comptime!(chunks.clone()),
                comptime!(StageStorage::Lanes { reach }),
                comptime!(None),
            );
            let mut sum = c_plane.cmma_accumulator::<E, E>(
                &a_plane,
                comptime!(Fragments::below(&c_plane, &a_plane)),
                Monoid::Sum,
            );
            sum.zero();
            for chunk in plane {
                lines.copy_from(&scale_plane.at(&chunk));
                for step in chunk {
                    // The step's window of each factor, landed once.
                    let a_step = a_plane
                        .at(&step)
                        .plain()
                        .landed(Side::Lhs, comptime!(out.clone()));
                    let b_step = b_plane
                        .at(&step)
                        .scaled(&ComptimeOption::new_Some(lines.at(&step)))
                        .landed(Side::Rhs, comptime!(out.clone()));
                    for block in step.walk().unrolled() {
                        for depth in block.walk().unrolled() {
                            let a_f = PlanePartition::<E>::cmma_fragments(&a_step.at(&depth), &sum);
                            let b_f = PlanePartition::<E>::cmma_fragments(&b_step.at(&depth), &sum);
                            for cell in depth.walk().unrolled() {
                                let mut sum_cell = sum.at(&cell);
                                sum_cell.mma(&a_f.at(&cell), &b_f.at(&cell), Semiring::SUM_PROD);
                            }
                        }
                    }
                }
            }
            sum.drained_into(&c_plane, comptime!(Some(grid.clone())));
        }
    }
}

/// **A tile-ordered weight lands on the tensor cores under a partition.** Sixteen rows: a plane
/// holds two fragments of rows by two of columns and walks the contraction two blocks a step,
/// two instructions a block, landing every step's window once.
fn check_partitioned(scales: TileScales, reach: Reach) {
    let (rows, n_tiles, chunk, chunks, fragment) = (16, 2, 32, 2, 8);
    let w = TileOrdered::new(rows, n_tiles, chunk * chunks);
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let c = TestInput::builder(client.clone(), shape![rows, w.cols()])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // Leaf up: the instruction; the grid of fragments a plane holds at one depth; the quant
    // block, two instructions deep under one scale; the partition's depth, two blocks a step;
    // the steps of a chunk; the chunks; the planes; the cubes.
    let levels = Tiling::leaf(&[(M, fragment), (NI, fragment), (KI, fragment)])
        .walk(&[(M, rows / fragment), (NI, w.tile / fragment)])
        .walk(&[(KI, w.tile / fragment)])
        .walk(&[(KB, 2)])
        .walk(&[(KB, chunk / 2)])
        .walk_every(&[KB])
        .planes(&[(NI, 1)])
        .cubes(&[NB])
        .levels();
    let chunks_level = levels[2].clone();
    let grid = levels[6].clone();
    let launcher = Launcher::implied(
        &client,
        Partitioning::new(w.space(), levels),
        KernelForm::Static,
    );
    let (s_op, stored) = w.s_op(&client, &launcher, scales);

    partitioned_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w.a_op(&client, &launcher).arg(),
        w.b_op(&client, &launcher).arg(),
        s_op.arg(),
        w.c_op(&launcher, &c).arg(),
        launcher.partitioning_arg(),
        chunks_level,
        reach,
        grid,
        [dtype, dtype, stored],
    );
    w.check(&client, c, &format!("partition {scales:?}"));
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores_under_a_partition() {
    for reach in [Reach::Shuffle, Reach::Window] {
        check_partitioned(TileScales::F32, reach);
    }
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores_under_a_partition_with_byte_scales() {
    for reach in [Reach::Shuffle, Reach::Window] {
        check_partitioned(TileScales::Ue4m3, reach);
    }
}
