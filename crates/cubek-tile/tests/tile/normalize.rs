//! Backend coverage for the three denominator addressing paths in `Tile::normalize`.

use cubecl::{Runtime, TestRuntime, ir::ElemType, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 2;
const COLS: usize = 4;

#[cube(launch)]
fn normalize_kernel<E: Float, M: Numeric, A: Size, D: Size>(
    accumulator: &TileArg<'_, E, A>,
    denominator: &TileArg<'_, M, D>,
    #[comptime] space: Space,
    #[comptime] guarded: bool,
    #[define(E)] _dtype: ElemType,
    #[define(M)] _meta_dtype: ElemType,
) {
    let mut accumulator = accumulator.tile(comptime!(space.clone()));
    let denominator = denominator.tile(space);
    let guard = comptime!(if guarded {
        DivGuard {
            epsilon: 1.0e-6,
            fallback: 0.25,
        }
    } else {
        DivGuard::default()
    });
    accumulator.normalize(&denominator, guard);
}

fn run_f32(
    values: Vec<f32>,
    denominators: Vec<f32>,
    denominator_shape: cubecl::zspace::Shape,
    denominator_axes: &[Axis],
    accumulator_width: usize,
    denominator_width: usize,
    guarded: bool,
) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = Space::new(&[(ROW, ROWS), (COL, COLS)]);
    let (accumulator, _) = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .custom(values)
        .generate_with_f32_host_data();
    let (denominator, _) = TestInput::builder(client.clone(), denominator_shape)
        .dtype(dtype)
        .custom(denominators)
        .generate_with_f32_host_data();

    normalize_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        accumulator_width,
        denominator_width,
        TileArgLaunch::new(
            accumulator.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        TileArgLaunch::new(
            denominator.binding().into_tensor_arg(),
            TileSpec::direct(denominator_axes),
        ),
        space,
        guarded,
        dtype,
        dtype,
    );

    HostData::from_tensor_handle(&client, accumulator, HostDataType::F32)
}

fn run_u32(values: Vec<f32>, denominators: Vec<f32>) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let f32_dtype = f32::elem_type_native();
    let u32_dtype = u32::elem_type_native();
    let space = Space::new(&[(ROW, ROWS), (COL, COLS)]);
    let (accumulator, _) = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(f32_dtype)
        .custom(values)
        .generate_with_f32_host_data();
    let denominator = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(u32_dtype)
        .custom(denominators)
        .generate_without_host_data();

    normalize_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        4,
        4,
        TileArgLaunch::new(
            accumulator.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        TileArgLaunch::new(
            denominator.binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        false,
        f32_dtype,
        u32_dtype,
    );

    HostData::from_tensor_handle(&client, accumulator, HostDataType::F32)
}

fn assert_values(got: &HostData, expected: &[f32]) {
    for (i, &want) in expected.iter().enumerate() {
        let have = got.get_f32(&[i / COLS, i % COLS]);
        assert!(
            (have - want).abs() < 1.0e-6,
            "normalization at {i}: got {have}, want {want}"
        );
    }
}

#[test]
fn normalization_uses_aligned_vector_denominators() {
    let got = run_f32(
        vec![8.0; ROWS * COLS],
        vec![2.0, 4.0, 8.0, 16.0, 1.0, 2.0, 4.0, 8.0],
        shape![ROWS, COLS],
        &[ROW, COL],
        4,
        4,
        false,
    );
    assert_values(&got, &[4.0, 2.0, 1.0, 0.5, 8.0, 4.0, 2.0, 1.0]);
}

#[test]
fn normalization_broadcasts_a_lane_invariant_denominator() {
    let got = run_f32(
        vec![8.0; ROWS * COLS],
        vec![2.0, 4.0],
        shape![ROWS],
        &[ROW],
        4,
        1,
        false,
    );
    assert_values(&got, &[4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0]);
}

#[test]
fn normalization_gathers_a_differently_lined_denominator() {
    let got = run_f32(
        vec![8.0; ROWS * COLS],
        vec![2.0, 4.0, 8.0, 16.0, 1.0, 2.0, 4.0, 8.0],
        shape![ROWS, COLS],
        &[ROW, COL],
        4,
        2,
        false,
    );
    assert_values(&got, &[4.0, 2.0, 1.0, 0.5, 8.0, 4.0, 2.0, 1.0]);
}

#[test]
fn normalization_guard_handles_zero_near_zero_negative_and_nan_on_device() {
    let got = run_f32(
        vec![8.0; ROWS * COLS],
        vec![0.0, 1.0e-8, -2.0, f32::NAN, 0.0, 1.0e-8, -2.0, f32::NAN],
        shape![ROWS, COLS],
        &[ROW, COL],
        4,
        4,
        true,
    );
    assert_values(&got, &[2.0, 2.0, -4.0, 2.0, 2.0, 2.0, -4.0, 2.0]);
}

#[test]
fn normalization_casts_an_integer_denominator() {
    let got = run_u32(
        vec![8.0; ROWS * COLS],
        vec![2.0, 4.0, 8.0, 16.0, 1.0, 2.0, 4.0, 8.0],
    );
    assert_values(&got, &[4.0, 2.0, 1.0, 0.5, 8.0, 4.0, 2.0, 1.0]);
}
