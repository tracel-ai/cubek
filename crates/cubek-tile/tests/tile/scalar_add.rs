use cubecl::{TestRuntime, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TileInput, assert_equals_approx,
};
use cubek_tile::{Axis, Space, Storage, TileArg, TileArgLaunch};

const M: Axis = Axis(0);
const N: Axis = Axis(1);

/// Base sanity: the kernel runs and adds the scalar on a small tile.
#[test]
fn scalar_add_kernel_works() {
    run_non_quantized(4, 4, 5.0);
}

/// Non-quantized path over a larger tile, with a negative scalar.
#[test]
fn scalar_add_non_quantized_matches_reference() {
    run_non_quantized(8, 8, -2.5);
}

/// RED (TDD): the first-class goal — calling `scalar_add` on a *quantized* input must
/// transparently dequantize on read, i.e. produce `q * scale + scalar`. `cubek-tile` can't
/// depend on `cubek-quant`, so the quantized input is modeled directly as a `u32` tensor of
/// integers plus a host-side `scale` (no packing). `scalar_add` currently reads the raw integer
/// and ignores the scale, so this FAILS until the quant-aware tile read applies it.
#[test]
fn scalar_add_quantized_matches_reference() {
    run_quantized(8, 8, -3.0);
}

/// Launch `scalar_add` over a plain (non-quantized) f32 tensor and check `out == in + scalar`.
fn run_non_quantized(m: usize, n: usize, scalar: f32) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let space = Space::new(&[(M, m), (N, n)]);

    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space).untiled().zeros();

    let dtype = f32::as_type_native_unchecked().storage_type();
    scalar_add::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(input.tensor_arg(1), input.space(), input.storage()),
        scalar,
        TileArgLaunch::new(output.tensor_arg(1), output.space(), output.storage()),
        dtype,
        dtype,
        1usize,
    );

    let input_host = HostData::from_tensor_handle(&client, input.handle(), HostDataType::F32);
    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);

    let shape = Shape::from(vec![m, n]);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| input_host.get_f32(&idx) + scalar)
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };

    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}

#[cube(launch)]
/// input: the input tensor
/// scalar: the scalar to multiply with
/// output: the output tensor
pub fn scalar_add<I: Numeric, O: Numeric, N: Size>(
    input: &TileArg<'_, I, N>,
    scalar: f32,
    output: &TileArg<'_, O, N>,
    #[define(I)] _input_dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
    #[define(N)] _size: usize,
) {
    let input = input.tile();
    let mut output = output.tile();
    output.add_scalar::<Vector<I, N>, f32>(&input, scalar);
}

/// Launch `scalar_add` over a quantized input and check `out == q * scale + scalar`.
fn run_quantized(m: usize, n: usize, scalar: f32) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let scale = 0.5f32;

    // Quantized integers stored in u32 (wgpu-compatible), plus a separate scale.
    let q: Vec<u32> = (0..(m * n) as u32).collect();
    let shape = Shape::from(vec![m, n]);
    let alloc = client.create_tensor_from_slice(u32::as_bytes(&q), shape.clone(), u32::type_size());
    let input = TensorHandle::<TestRuntime>::new(
        alloc.memory,
        shape.clone(),
        alloc.strides,
        u32::as_type_native_unchecked(),
    );

    let space = Space::new(&[(M, m), (N, n)]);
    let storage = Storage::of(2, 2);
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    let in_dtype = u32::as_type_native_unchecked().storage_type();
    let out_dtype = f32::as_type_native_unchecked().storage_type();
    scalar_add::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(input.binding().into_tensor_arg(), space, storage),
        scalar,
        TileArgLaunch::new(output.tensor_arg(1), output.space(), output.storage()),
        in_dtype,
        out_dtype,
        1usize,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(q.iter().map(|&v| v as f32 * scale + scalar).collect()),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };

    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}
