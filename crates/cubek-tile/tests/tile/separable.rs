//! A separable procedural weight contracted over three axes: the rank the abstraction carries is
//! the recipe's, not the microkernel's.
//!
//! `out[row, col] = Σ_{k0, k1, k2} (∏_f factor_f(k_f, row)) · in[k0, k1, k2, col]`. The same
//! recipe is run twice, once stating its factorization and once opaque, so the cell-major walk is
//! checked against the general one as well as against the host.
#![allow(non_snake_case)]

use cubecl::{Runtime, TestRuntime, features::TypeUsage, ir::ElemType, prelude::*, zspace::shape};
use cubek_quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_test_utils::{
    HostData, HostDataType, TestInput, TestOutcome, TileInput, ValidationResult,
};
use cubek_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const TAP: [Axis; 3] = [Axis(2), Axis(3), Axis(4)];

const ROWS: usize = 2;
const COLS: usize = 3;
/// Deliberately unequal, so a factor read at the wrong offset lands on the wrong tap count.
const TAPS: [usize; 3] = [2, 3, 2];

/// `offset + coefficient * tap + row_coefficient * row`, per factor. Only the first factor reads
/// the output row, which is the shape a resampling weight takes: a factor may read a free axis,
/// it may only not read another factor's contracted axis.
const OFFSET: [f32; 3] = [1.0, 2.0, 3.0];
const COEFFICIENT: [f32; 3] = [1.0, 3.0, 5.0];
const ROW_COEFFICIENT: [f32; 3] = [2.0, 0.0, 0.0];

/// Every factor is the same type, which is what a [`SeparableProduct`] holds: one filter family
/// applied along each axis, at that axis's own coordinates.
type Factor<E> = Sum<AffineCoordinate<E>, AffineCoordinate<E>>;
type Weights<E> = SeparableProduct<Factor<E>>;

fn factor_value(f: usize, tap: usize, row: usize) -> f32 {
    OFFSET[f] + COEFFICIENT[f] * tap as f32 + ROW_COEFFICIENT[f] * row as f32
}

#[cube]
fn factor<E: Float>(#[comptime] f: usize) -> Factor<E> {
    sum_of(
        affine_along(
            comptime!(TAP[f]),
            E::new(comptime!(OFFSET[f])),
            E::new(comptime!(COEFFICIENT[f])),
        ),
        affine_along(ROW, E::new(0.0_f32), E::new(comptime!(ROW_COEFFICIENT[f]))),
    )
}

#[cube]
fn weights<E: Float>() -> Weights<E> {
    let mut factors = Sequence::new();
    #[unroll]
    for f in 0..comptime!(TAP.len()) {
        factors.push(factor::<E>(f));
    }
    separable_product(factors)
}

#[cube(launch)]
fn separable_kernel<E: Float>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] separable: bool,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weight_axes = comptime!([&[ROW], TAP.as_slice()].concat());
    let weights = if comptime!(separable) {
        Tile::<E>::procedural_separable::<Weights<E>>(
            comptime!(space.project(&weight_axes)),
            weights::<E>(),
        )
    } else {
        Tile::<E>::procedural::<Weights<E>>(comptime!(space.project(&weight_axes)), weights::<E>())
    };

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}

/// Small integers, so the accumulation is exact in `f32`.
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 7) as f32) - 2.0).collect()
}

fn reference(input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        for col in 0..COLS {
            let mut acc = 0.0f32;
            for k0 in 0..TAPS[0] {
                for k1 in 0..TAPS[1] {
                    for k2 in 0..TAPS[2] {
                        let weight = factor_value(0, k0, row)
                            * factor_value(1, k1, row)
                            * factor_value(2, k2, row);
                        let at = ((k0 * TAPS[1] + k1) * TAPS[2] + k2) * COLS + col;
                        acc += weight * input[at];
                    }
                }
            }
            out[row * COLS + col] = acc;
        }
    }
    out
}

fn run(separable: bool) -> (HostData, Vec<f32>) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    let in_shape = shape![TAPS[0], TAPS[1], TAPS[2], COLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[
            (ROW, ROWS),
            (COL, COLS),
            (TAP[0], TAPS[0]),
            (TAP[1], TAPS[1]),
            (TAP[2], TAPS[2]),
        ])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(ROWS))
                .axis(COL, Cut::sequential(COLS))
                .axis(TAP[0], Cut::sequential(TAPS[0]))
                .axis(TAP[1], Cut::sequential(TAPS[1]))
                .axis(TAP[2], Cut::sequential(TAPS[2]))
        })
        .build();

    separable_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            in_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[TAP[0], TAP[1], TAP[2], COL]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        separable,
        space,
        f32_ty,
    );

    (
        HostData::from_tensor_handle(&client, out_handle, HostDataType::F32),
        in_data,
    )
}

fn check(separable: bool) {
    let (got, input) = run(separable);
    let want = reference(&input);
    for row in 0..ROWS {
        for col in 0..COLS {
            let have = got.get_f32(&[row, col]);
            let want = want[row * COLS + col];
            assert!(
                (have - want).abs() < 1e-4,
                "separable {separable}: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

/// Three factors, walked one axis at a time: `2 + 3 + 2` evaluations per cell rather than
/// `2 * 3 * 2`, and the same values.
#[test]
fn a_rank_three_separable_product_contracts_factor_by_factor() {
    check(true);
}

/// The same recipe with its factorization withheld, contracted the general way. A recipe is the
/// product of its factors, so both paths must agree.
#[test]
fn an_opaque_product_contracts_to_the_same_values() {
    check(false);
}

#[test]
fn a_separable_lhs_contracts_a_padded_staged_rhs() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    let in_shape = shape![TAPS[0], TAPS[1], TAPS[2], COLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[
            (ROW, ROWS),
            (COL, COLS),
            (TAP[0], TAPS[0]),
            (TAP[1], TAPS[1]),
            (TAP[2], TAPS[2]),
        ])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(ROWS))
                .axis(COL, Cut::sequential(COLS))
                .axis(TAP[0], Cut::sequential(TAPS[0]))
                .axis(TAP[1], Cut::sequential(TAPS[1]))
                .axis(TAP[2], Cut::sequential(TAPS[2]))
        })
        .build();

    let in_spec = TileSpec::direct(&[TAP[0], TAP[1], TAP[2], COL])
        .residence(&[Residence::Smem])
        .stage_width(4);

    separable_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        true,
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let want = reference(&in_data);
    for row in 0..ROWS {
        for col in 0..COLS {
            let have = got.get_f32(&[row, col]);
            let want = want[row * COLS + col];
            assert!(
                (have - want).abs() < 1e-4,
                "separable padded staged: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

// ---- separable lhs against a quantized rhs ---------------------------------

/// A quantized rhs is read through a dequantizing view over its *storage* buffer, which the view
/// reinterprets at `served / pack` elements per line. The two cases below sit on either side of
/// that ratio: packed-u32 serves exactly one storage word per line, native serves `QV` of them,
/// and only the second tells a correct width apart from a hardcoded scalar one.
const QCOLS: usize = 4;
const QV: usize = 4;
const QSCALE: f32 = 0.05;

#[cube(launch)]
fn separable_quant_kernel<E: Float, I: Numeric, VI: Size, V: Size>(
    input: &QuantTileArg<'_, I, VI>,
    output: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[define(I)] _input_dtype: ElemType,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile::<E>(comptime!(space.clone()));
    let weight_axes = comptime!([&[ROW], TAP.as_slice()].concat());
    let weights = Tile::<E>::procedural_separable::<Weights<E>>(
        comptime!(space.project(&weight_axes)),
        weights::<E>(),
    );

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}

/// Native Q8S served in `QV`-wide lines: `served / pack` is `QV`, so a scalar physical width
/// would walk the storage buffer one element at a time and serve the wrong `q` per column. The
/// case that discriminates the width, and the one a backend without native i8 cannot run.
#[test]
fn a_separable_lhs_contracts_a_native_quantized_rhs() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }
    let max_width = client.properties().hardware.max_vector_size;
    if QV > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below the {QV}-wide served line"
        )))
        .enforce();
        return;
    }

    // Per-tensor, so one scale covers every line and nothing can straddle a block.
    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let in_shape = shape![TAPS[0], TAPS[1], TAPS[2], QCOLS];
    let in_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (in_handle, in_host) = TestInput::builder(client.clone(), in_shape)
        .dtype(in_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![1])
        .custom(vec![QSCALE])
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[
            (ROW, ROWS),
            (COL, QCOLS),
            (TAP[0], TAPS[0]),
            (TAP[1], TAPS[1]),
            (TAP[2], TAPS[2]),
        ])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(ROWS))
                .axis(COL, Cut::sequential(QCOLS))
                .axis(TAP[0], Cut::sequential(TAPS[0]))
                .axis(TAP[1], Cut::sequential(TAPS[1]))
                .axis(TAP[2], Cut::sequential(TAPS[2]))
        })
        .build();

    let launcher = space.launcher_over(&client, &[]);
    let input_op = launcher
        .arg(in_handle.binding())
        .subspace(&[TAP[0], TAP[1], TAP[2], COL])
        .vectorize(QV)
        .quantized(&[scales.binding()], scheme, DequantAt::Read)
        .build();

    let f32_ty = f32::elem_type_native();
    let out_handle = TestInput::builder(client.clone(), shape![ROWS, QCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    separable_quant_kernel::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        input_op.bound_width(),
        QV,
        input_op.arg(),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.space().clone(),
        in_dtype,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..QCOLS {
            let mut want = 0.0f32;
            for k0 in 0..TAPS[0] {
                for k1 in 0..TAPS[1] {
                    for k2 in 0..TAPS[2] {
                        let weight = factor_value(0, k0, row)
                            * factor_value(1, k1, row)
                            * factor_value(2, k2, row);
                        want += weight * in_host.get_f32(&[k0, k1, k2, col]) * QSCALE;
                    }
                }
            }
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-3,
                "separable quantized: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

/// Packed-u32 Q8S: one storage word per served line, so this runs on every backend and covers the
/// separable walk against a dequantizing view end to end, scales included.
#[test]
fn a_separable_lhs_contracts_a_packed_quantized_rhs() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below the packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let space = Tiling::new()
        .extents(&[
            (ROW, ROWS),
            (COL, pack),
            (TAP[0], TAPS[0]),
            (TAP[1], TAPS[1]),
            (TAP[2], TAPS[2]),
        ])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(ROWS))
                .axis(COL, Cut::sequential(pack))
                .axis(TAP[0], Cut::sequential(TAPS[0]))
                .axis(TAP[1], Cut::sequential(TAPS[1]))
                .axis(TAP[2], Cut::sequential(TAPS[2]))
        })
        .build();

    let input = TileInput::builder(&client, space.project(&[TAP[0], TAP[1], TAP[2], COL]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();

    let f32_ty = f32::elem_type_native();
    let out_handle = TestInput::builder(client.clone(), shape![ROWS, pack])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = space.launcher_over(&client, &[]);
    let input_op = launcher
        .arg(input.tile.handle().binding())
        .subspace(&[TAP[0], TAP[1], TAP[2], COL])
        .vectorize(pack)
        .quantized(&[input.scales_binding()], scheme, DequantAt::Read)
        .build();

    separable_quant_kernel::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        input_op.bound_width(),
        pack,
        input_op.arg(),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.space().clone(),
        u32::elem_type_native(),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let scale = input.scale_values[0];
    for row in 0..ROWS {
        for col in 0..pack {
            let mut want = 0.0f32;
            for k0 in 0..TAPS[0] {
                for k1 in 0..TAPS[1] {
                    for k2 in 0..TAPS[2] {
                        let weight = factor_value(0, k0, row)
                            * factor_value(1, k1, row)
                            * factor_value(2, k2, row);
                        let at = ((k0 * TAPS[1] + k1) * TAPS[2] + k2) * pack + col;
                        want += weight * input.q[at] as f32 * scale;
                    }
                }
            }
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-3,
                "separable packed: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

// ---- separable lhs against a resampling rhs --------------------------------

/// The rhs's one gathered physical axis: `⌊(3·row + 2·tap) / 2⌋`, which is `⌊3·row / 2⌋ + tap`.
///
/// Both halves of the split the separable schedule runs on are load-bearing here. `row` stays
/// inside the floor, so it has to be anchored; `tap` has a coefficient the divisor factors out, so
/// it is stepped by `1` on top of that anchor. A schedule folding the whole map per tap would get
/// the same answer, which is the point: this pins the hand-folded one against it.
const ROW_NUM: usize = 3;
const RESAMPLE: usize = 2;
const RTAPS: usize = 2;
const RROWS: usize = 4;
const RCOLS: usize = 3;

fn resample_origin(row: usize) -> usize {
    (ROW_NUM * row) / RESAMPLE
}

/// One factor, over the single contracted axis: a stated factorization of rank one still takes the
/// separable schedule, since the per-row weight walk it caches is worth `nr` evaluations whatever
/// the rank.
#[cube]
fn resample_weights<E: Float>() -> Weights<E> {
    let mut factors = Sequence::new();
    factors.push(factor::<E>(0usize));
    separable_product(factors)
}

#[cube(launch)]
fn resample_kernel<E: Float>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] normalized: bool,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weights = Tile::<E>::procedural_separable::<Weights<E>>(
        comptime!(space.project(&[ROW, TAP[0]])),
        resample_weights::<E>(),
    );
    let weights = if comptime!(normalized) {
        weights.normalized(comptime!(TapMask::Unmasked), comptime!(DivGuard::default()))
    } else {
        weights
    };

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}

#[test]
fn a_separable_lhs_contracts_a_resampling_rhs() {
    check_resampling(false);
}

#[test]
fn a_separable_resampling_lhs_normalizes_its_factor_run() {
    check_resampling(true);
}

fn check_resampling(normalized: bool) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    let in_rows = resample_origin(RROWS - 1) + RTAPS;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(RROWS))
                .axis(COL, Cut::sequential(RCOLS))
                .axis(TAP[0], Cut::sequential(RTAPS))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ));

    resample_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        normalized,
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut want = 0.0f32;
            for tap in 0..RTAPS {
                let at = (resample_origin(row) + tap) * RCOLS + col;
                want += factor_value(0, tap, row) * in_data[at];
            }
            if normalized {
                want /= (0..RTAPS).map(|tap| factor_value(0, tap, row)).sum::<f32>();
            }
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "separable resample (normalized={normalized}): at ({row}, {col}) got {have}, \
                 want {want}"
            );
        }
    }
}

// ---- masked normalization against a procedural trailing tile ----------------

/// Normalize one deliberately child-local factor run at a time. This shape makes the rhs's
/// second procedural window contain one real tap and one padded tap, so `TapMask::Masked` must
/// distinguish the checked-read zero from an in-bounds sample without relying on backing memory.
#[cube(launch)]
fn procedural_mask_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let rhs = Tile::<E>::procedural::<AffineCoordinate<E>>(
        comptime!(space.project(&[TAP[0], COL])),
        affine_along(TAP[0], E::new(1.0_f32), E::new(1.0_f32)),
    );
    let mut output = output.tile(comptime!(space.clone()));
    output.zero();

    for region in Walk::over(rhs.runtime_space()) {
        let rhs = rhs.at(&region);
        let child = comptime!(space.divide());
        let mut factors = Sequence::new();
        factors.push(affine_along(TAP[0], E::new(1.0_f32), E::new(0.0_f32)));
        let weights = Tile::<E>::procedural_separable::<SeparableProduct<AffineCoordinate<E>>>(
            comptime!(child.project(&[ROW, TAP[0]])),
            separable_product(factors),
        )
        .normalized(comptime!(TapMask::Masked), comptime!(DivGuard::default()));
        output.at(&region).mma(&weights, &rhs, Semiring::SUM_PROD);
    }
}

#[test]
fn masked_normalization_excludes_a_procedural_overhang() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![1, 1])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let space = Tiling::new()
        .extents(&[(ROW, 1), (COL, 1), (TAP[0], 3)])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(1))
                .axis(COL, Cut::sequential(1))
                .axis(TAP[0], Cut::sequential(2))
        })
        .build();

    procedural_mask_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32).get_f32(&[0, 0]);
    // Mechanism assertion: tests that `TapMask::Masked` excludes the overhang tap across chunks.
    // First chunk: (1 + 2) / 2. Trailing chunk: 3 / 1 (with its padded fourth tap excluded).
    // Sum across chunks yields 1.5 + 3.0 = 4.5.
    assert!((got - 4.5).abs() < 1.0e-6, "got {got}, want 4.5");
}

// ---- masked normalization against Gmem Boundary::Zero input -----------------

#[cube(launch)]
fn resample_kernel_masked<E: Float>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weights = Tile::<E>::procedural_separable::<Weights<E>>(
        comptime!(space.project(&[ROW, TAP[0]])),
        resample_weights::<E>(),
    )
    .normalized(comptime!(TapMask::Masked), comptime!(DivGuard::default()));

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}

#[test]
fn masked_normalization_dedarkens_a_boundary_zero_gmem_input() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    // Deliberately clip input rows so the last output row's tap window overhangs the edge.
    let in_rows = resample_origin(RROWS - 1) + 1;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(RROWS))
                .axis(COL, Cut::sequential(RCOLS))
                .axis(TAP[0], Cut::sequential(RTAPS))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ))
    .checked(true);

    resample_kernel_masked::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for tap in 0..RTAPS {
                let in_r = resample_origin(row) + tap;
                if in_r < in_rows {
                    let w = factor_value(0, tap, row);
                    num += w * in_data[in_r * RCOLS + col];
                    den += w;
                }
            }
            let want = if den != 0.0 { num / den } else { 0.0 };
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "masked gmem resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

/// The staged twin of [`masked_normalization_dedarkens_a_boundary_zero_gmem_input`].
///
/// `TapMask::Masked` has to drop the taps that overhang the input, and the fill has already
/// replaced those with zeros by the time the leaf reads them: a staged window cannot tell a padded
/// zero from a real sample. The stage therefore records the window it was filled from, and the
/// mask is put to that rectangle instead. The expected values are the gmem test's, because staging
/// is a placement decision and must not move a number.
#[test]
fn masked_normalization_dedarkens_a_boundary_zero_smem_input() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    // Same clipped input as the gmem twin, so the last output row's taps overhang the edge.
    let in_rows = resample_origin(RROWS - 1) + 1;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(RROWS))
                .axis(COL, Cut::sequential(RCOLS))
                .axis(TAP[0], Cut::sequential(RTAPS))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ))
    .checked(true)
    .residence(&[Residence::Smem]);

    resample_kernel_masked::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let mut overhanging = 0usize;
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for tap in 0..RTAPS {
                let in_r = resample_origin(row) + tap;
                if in_r < in_rows {
                    let w = factor_value(0, tap, row);
                    num += w * in_data[in_r * RCOLS + col];
                    den += w;
                } else if col == 0 {
                    overhanging += 1;
                }
            }
            let want = if den != 0.0 { num / den } else { 0.0 };
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "masked smem resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
    // Without an overhanging tap the mask is a no-op and the test would pass on a stage that
    // dropped the boundary entirely, which is the bug this defends.
    assert!(
        overhanging > 0,
        "masked smem resample: no tap overhangs the input, so the mask was never exercised"
    );
}

// ---- column-spanning normalized separable contraction -----------------------

#[cube(launch)]
fn column_spanning_resample_kernel<E: Float>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weights = Tile::<E>::procedural_separable::<Weights<E>>(
        comptime!(space.project(&[ROW, COL, TAP[0]])),
        resample_weights::<E>(),
    )
    .normalized(comptime!(TapMask::Unmasked), comptime!(DivGuard::default()));

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}

#[test]
fn a_column_spanning_separable_lhs_normalizes_its_factor_run() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    // `Weights` only reads `TAP[0]` and `ROW`, so adding `COL` to the LHS space isolates the
    // column-spanning coordinate schedule without changing the mathematical values.
    let in_rows = resample_origin(RROWS - 1) + RTAPS;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(RROWS))
                .axis(COL, Cut::sequential(RCOLS))
                .axis(TAP[0], Cut::sequential(RTAPS))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ));

    column_spanning_resample_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut want = 0.0f32;
            for tap in 0..RTAPS {
                let at = (resample_origin(row) + tap) * RCOLS + col;
                want += factor_value(0, tap, row) * in_data[at];
            }
            want /= (0..RTAPS).map(|tap| factor_value(0, tap, row)).sum::<f32>();
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "column spanning separable resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

#[cube(launch)]
fn column_spanning_resample_kernel_masked<E: Float>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let weights = Tile::<E>::procedural_separable::<Weights<E>>(
        comptime!(space.project(&[ROW, COL, TAP[0]])),
        resample_weights::<E>(),
    )
    .normalized(comptime!(TapMask::Masked), comptime!(DivGuard::default()));

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}

#[test]
fn a_column_spanning_separable_lhs_masks_and_dedarkens_boundary_zero_gmem_input() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    // Deliberately clip input rows so trailing output rows overhang the Boundary::Zero edge.
    let in_rows = resample_origin(RROWS - 1) + 1;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(RROWS))
                .axis(COL, Cut::sequential(RCOLS))
                .axis(TAP[0], Cut::sequential(RTAPS))
        })
        .build();

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ))
    .checked(true);

    column_spanning_resample_kernel_masked::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for tap in 0..RTAPS {
                let in_r = resample_origin(row) + tap;
                if in_r < in_rows {
                    let w = factor_value(0, tap, row);
                    num += w * in_data[in_r * RCOLS + col];
                    den += w;
                }
            }
            let want = if den != 0.0 { num / den } else { 0.0 };
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "column spanning masked gmem resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

// ---- zero factor sum taking fallback without poisoning siblings -------------

#[cube(launch)]
fn zero_sum_fallback_kernel<E: Float>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let mut factors = Sequence::new();
    // Factor 0: taps at k=0 (1.0) and k=1 (-1.0), sum = 0.0
    factors.push(affine_along(TAP[0], E::new(1.0_f32), E::new(-2.0_f32)));
    // Factor 1: taps at k=0 (2.0) and k=1 (2.0), sum = 4.0
    factors.push(affine_along(TAP[1], E::new(2.0_f32), E::new(0.0_f32)));
    let weights = Tile::<E>::procedural_separable::<SeparableProduct<AffineCoordinate<E>>>(
        comptime!(space.project(&[ROW, TAP[0], TAP[1]])),
        separable_product(factors),
    )
    .normalized(
        comptime!(TapMask::Unmasked),
        comptime!(DivGuard {
            epsilon: 1.0e-7,
            fallback: 3.0,
        }),
    );

    let mut output = output.tile(space);
    output.mm(&weights, &input, Semiring::SUM_PROD);
}

#[test]
fn a_zero_factor_sum_takes_fallback_without_poisoning_siblings() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_ty = f32::elem_type_native();

    // Varies along TAP[0] so factor 0's antisymmetric taps do not cancel: the result then pins
    // the fallback value itself, not just the absence of a NaN.
    let in_handle = TestInput::builder(client.clone(), shape![2, 2, 1])
        .dtype(f32_ty)
        .custom(vec![1.0f32, 1.0, 2.0, 2.0])
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![1, 1])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let space = Tiling::new()
        .extents(&[(ROW, 1), (COL, 1), (TAP[0], 2), (TAP[1], 2)])
        .instruction(Instruction::registers(16), |l| {
            l.axis(ROW, Cut::sequential(1))
                .axis(COL, Cut::sequential(1))
                .axis(TAP[0], Cut::sequential(2))
                .axis(TAP[1], Cut::sequential(2))
        })
        .build();

    zero_sum_fallback_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            in_handle.binding().into_tensor_arg(),
            TileSpec::direct(&[TAP[0], TAP[1], COL]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32).get_f32(&[0, 0]);
    // Factor 0 has sum 0.0 -> fallback recip 3.0 -> taps become [3.0, -3.0].
    // Factor 1 has sum 4.0 -> recip 0.25 -> taps become [0.5, 0.5].
    // Product sum = (3.0 * 1.0 - 3.0 * 2.0) * (0.5 + 0.5) = -3.0.
    assert!((got + 3.0).abs() < 1.0e-6, "got {got}, want -3.0");
}
