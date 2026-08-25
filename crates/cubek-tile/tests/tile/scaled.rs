//! `c.mm_scaled(&a, &b, &s)`: the contraction with one operand scaled by a **real operand**.
//!
//! *Which* operand is not stated: the scales' own axes say it. A scale over the output's columns
//! is a fact about the rhs's columns and nothing else could fold it in; anything else scales the
//! lhs. One verb, then, and the same kernel body serves both — `(a ⊗ s) · b` and `a · (b ⊗ s)` are
//! the same sum of terms, differing only in where the factor folds in cheapest.
//!
//! The point is what the kernel signature says. The values are a tensor of values, the scales are
//! a tensor of scales, both are named at the call, and the arithmetic that folds one into the
//! other is a verb the kernel writes. Nothing decodes behind a read and nothing rides a binding's
//! side channel — so the scales' element type is just the type of the tensor bound, which is why
//! `f16` scales work here without a widening pass anywhere.
//!
//! The scales resolve at their own granularity through their own projection: `⌊k / BLOCK⌋` for a
//! per-block scale ([`coarse`](super::coarse)), a plain axis for a per-element one, an omitted
//! axis for a broadcast.

use cubecl::{Runtime, TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;
use half::f16;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const ROWS: usize = 4;
const COLS: usize = 4;
const DEPTH: usize = 32;
/// Contracted values per scale: the quantization block.
const BLOCK: usize = 8;
const BLOCKS: usize = DEPTH / BLOCK;

/// `c = (a ⊗ s) · b`, with `s` one value per `(row, block of K)`.
#[cube(launch)]
fn scaled_matmul<E: Numeric, S: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scales: &TileArg<'_, S, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.mm_scaled(&a, &b, &scales, Semiring::SUM_PROD);
}

/// [`scaled_matmul`] with the accumulator promoted to registers: the partials never round-trip
/// through `c`'s element between `K` steps, which is the form a decode gemv wants.
#[cube(launch)]
fn scaled_matmul_promoted<E: Numeric, S: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    scales: &TileArg<'_, S, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E, S)] _dtypes: [ElemType; 2],
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let scales = scales.tile(comptime!(space.clone()));
    let c = c.tile(space);
    let mut acc = c.accumulate::<E, _>(&a, Monoid::Sum);
    acc.mm_scaled(&a, &b, &scales, Semiring::SUM_PROD);
}

/// `⌊k / BLOCK⌋` along the contracted axis: one scale per block of the lhs's row.
fn scales_spec() -> TileSpec {
    TileSpec::new(Projection::new(
        &[M, K],
        &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(K).over(BLOCK)],
    ))
}

/// The rhs twin: one scale per `(block of K, column)`. Spanning `N` is what makes it the rhs's.
fn rhs_scales_spec() -> TileSpec {
    TileSpec::new(Projection::new(
        &[K, N],
        &[PhysicalAxisMap::of(K).over(BLOCK), PhysicalAxisMap::of(N)],
    ))
}

fn space(cut: usize) -> Space {
    Tiling::new()
        .extents(&[(M, ROWS), (N, COLS), (K, DEPTH)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(ROWS))
                .axis(N, Cut::sequential(COLS))
                .axis(K, Cut::sequential(cut))
        })
        .build()
        .with_instruction(Instruction::registers(16))
}

/// Small integers, so the reference is exact in `f32` and representable in `f16`.
fn lhs_data() -> Vec<f32> {
    (0..ROWS * DEPTH).map(|i| (i % 5) as f32 - 2.0).collect()
}

fn rhs_data() -> Vec<f32> {
    (0..DEPTH * COLS).map(|i| (i % 7) as f32 - 3.0).collect()
}

/// Distinct per `(row, block)`, and halves so an f16 scale is exact.
fn scale_data() -> Vec<f32> {
    (0..ROWS * BLOCKS).map(|i| (i as f32 + 1.0) / 2.0).collect()
}

/// Distinct per `(block, column)`, halves for the same reason.
fn rhs_scale_data() -> Vec<f32> {
    (0..BLOCKS * COLS).map(|i| (i as f32 + 1.0) / 2.0).collect()
}

/// `C[m,n] = Σ_k A[m,k] · S[m, k/BLOCK] · B[k,n]`.
fn reference() -> Vec<f32> {
    let (a, b, s) = (lhs_data(), rhs_data(), scale_data());
    let mut out = vec![0.0; ROWS * COLS];
    for m in 0..ROWS {
        for n in 0..COLS {
            out[m * COLS + n] = (0..DEPTH)
                .map(|k| a[m * DEPTH + k] * s[m * BLOCKS + k / BLOCK] * b[k * COLS + n])
                .sum();
        }
    }
    out
}

/// Launch [`scaled_matmul`] with the scales bound at `scale_dtype`.
fn run(space: Space, scale_dtype: ElemType) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

    let (a, _) = TestInput::builder(client.clone(), shape![ROWS, DEPTH])
        .dtype(dtype)
        .custom(lhs_data())
        .generate_with_f32_host_data();
    let (b, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(rhs_data())
        .generate_with_f32_host_data();
    let (scales, _) = TestInput::builder(client.clone(), shape![ROWS, BLOCKS])
        .dtype(scale_dtype)
        .custom(scale_data())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    scaled_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(a.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        TileArgLaunch::new(b.binding().into_tensor_arg(), TileSpec::direct(&[K, N])),
        TileArgLaunch::new(scales.binding().into_tensor_arg(), scales_spec()),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        [dtype, scale_dtype],
    );

    HostData::from_tensor_handle(&client, c, HostDataType::F32)
}

fn assert_scaled(got: &HostData) {
    let want = reference();
    for m in 0..ROWS {
        for n in 0..COLS {
            let have = got.get_f32(&[m, n]);
            let want = want[m * COLS + n];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The walk cuts `K` at the block, so each region carries one scale per row.
#[test]
fn a_scaled_contraction_folds_the_block_scale_in() {
    assert_scaled(&run(space(BLOCK), f32::elem_type_native()));
}

/// A cut finer than the block: several regions share a scale.
#[test]
fn a_cut_finer_than_the_block_reuses_its_scale() {
    assert_scaled(&run(space(BLOCK / 2), f32::elem_type_native()));
}

/// A cut coarser than the block: the scale changes within a region, so the step's own coordinate
/// is what addresses it.
#[test]
fn a_cut_coarser_than_the_block_changes_scale_within_a_region() {
    assert_scaled(&run(space(BLOCK * 2), f32::elem_type_native()));
}

/// **The one that pays for the design.** The scales are an `f16` tensor and the kernel reads
/// them as one — no widening pass, no scheme saying otherwise, no way for the two to disagree.
/// A scale is whatever its tensor holds because it is a tensor.
#[test]
fn f16_scales_are_read_as_f16() {
    assert_scaled(&run(space(BLOCK), f16::elem_type_native()));
}

/// `C[m,n] = Σ_k A[m,k] · B[k,n] · S[k/BLOCK, n]`: the rhs twin of [`reference`].
fn rhs_reference() -> Vec<f32> {
    let (a, b, s) = (lhs_data(), rhs_data(), rhs_scale_data());
    let mut out = vec![0.0; ROWS * COLS];
    for m in 0..ROWS {
        for n in 0..COLS {
            out[m * COLS + n] = (0..DEPTH)
                .map(|k| a[m * DEPTH + k] * b[k * COLS + n] * s[(k / BLOCK) * COLS + n])
                .sum();
        }
    }
    out
}

/// [`run`] with the scales spanning the rhs's axes instead: same kernel, same verb, and only the
/// operand's own projection differs.
fn run_rhs(space: Space) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

    let (a, _) = TestInput::builder(client.clone(), shape![ROWS, DEPTH])
        .dtype(dtype)
        .custom(lhs_data())
        .generate_with_f32_host_data();
    let (b, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(rhs_data())
        .generate_with_f32_host_data();
    let (scales, _) = TestInput::builder(client.clone(), shape![BLOCKS, COLS])
        .dtype(dtype)
        .custom(rhs_scale_data())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    scaled_matmul::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(a.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        TileArgLaunch::new(b.binding().into_tensor_arg(), TileSpec::direct(&[K, N])),
        TileArgLaunch::new(scales.binding().into_tensor_arg(), rhs_scales_spec()),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        space,
        [dtype, dtype],
    );

    HostData::from_tensor_handle(&client, c, HostDataType::F32)
}

fn assert_rhs_scaled(got: &HostData) {
    let want = rhs_reference();
    for m in 0..ROWS {
        for n in 0..COLS {
            let have = got.get_f32(&[m, n]);
            let want = want[m * COLS + n];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The scales span `N`, so they scale the rhs. Nothing at the call site says so: the verb reads
/// it off the operand.
#[test]
fn scales_over_the_columns_scale_the_rhs() {
    assert_rhs_scaled(&run_rhs(space(BLOCK)));
}

/// The rhs scale under a cut finer than its block: several regions share it.
#[test]
fn an_rhs_scale_survives_a_finer_cut() {
    assert_rhs_scaled(&run_rhs(space(BLOCK / 2)));
}

/// And under a coarser one, where the scale changes within a region.
#[test]
fn an_rhs_scale_changes_within_a_coarser_region() {
    assert_rhs_scaled(&run_rhs(space(BLOCK * 2)));
}

/// [`run`] with the output stating [`Residence::Register`]: the accumulator is a register block
/// living across the whole walk, and the scaled steps fold into it directly.
fn run_promoted(space: Space) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();

    let (a, _) = TestInput::builder(client.clone(), shape![ROWS, DEPTH])
        .dtype(dtype)
        .custom(lhs_data())
        .generate_with_f32_host_data();
    let (b, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(rhs_data())
        .generate_with_f32_host_data();
    let (scales, _) = TestInput::builder(client.clone(), shape![ROWS, BLOCKS])
        .dtype(dtype)
        .custom(scale_data())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // One entry per level: Register at the outermost, in place below it.
    let mut residence = vec![Residence::InPlace; space.partitioner().depth()];
    residence[0] = Residence::Register;

    scaled_matmul_promoted::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(a.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        TileArgLaunch::new(b.binding().into_tensor_arg(), TileSpec::direct(&[K, N])),
        TileArgLaunch::new(scales.binding().into_tensor_arg(), scales_spec()),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]).residence(&residence),
        ),
        space,
        [dtype, dtype],
    );

    HostData::from_tensor_handle(&client, c, HostDataType::F32)
}

/// **The gap the decode gemv was waiting on.** A promoted register accumulator under a scaled
/// contraction: refused before, and the same numbers as the memory-backed form.
#[test]
fn a_promoted_accumulator_takes_the_scaled_contraction() {
    assert_scaled(&run_promoted(space(BLOCK)));
}
