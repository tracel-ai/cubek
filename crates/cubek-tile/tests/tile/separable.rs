//! A separable procedural weight contracted over three axes: the rank the abstraction carries is
//! the recipe's, not the microkernel's.
//!
//! `out[row, col] = Σ_{k0, k1, k2} (∏_f factor_f(k_f, row)) · in[k0, k1, k2, col]`. The same
//! recipe is run twice, once stating its factorization and once opaque, so the cell-major walk is
//! checked against the general one as well as against the host.
#![allow(non_snake_case)]

use cubecl::{Runtime, TestRuntime, ir::ElemType, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, MEMORY_LEAF, TestInput};
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
    let leaf = comptime!(output.spec.leaf);
    let weight_axes = comptime!([&[ROW], TAP.as_slice()].concat());
    let weights = if comptime!(separable) {
        Tile::<E>::procedural_separable::<Weights<E>>(
            comptime!(space.project(&weight_axes)),
            weights::<E>(),
            leaf,
        )
    } else {
        Tile::<E>::procedural::<Weights<E>>(
            comptime!(space.project(&weight_axes)),
            weights::<E>(),
            leaf,
        )
    };

    let mut output = output.tile(space);
    output.zero();
    output.mma(&weights, &input);
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
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
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
            TileSpec::direct(&[TAP[0], TAP[1], TAP[2], COL], MEMORY_LEAF),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL], MEMORY_LEAF),
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
