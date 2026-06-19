use cubecl::{TestRuntime, ir::ElemType, prelude::*, zspace::Shape};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TestInput, TileInput, assert_equals_approx,
};
use cubek_tile::{
    Axis, QuantArgLaunch, QuantTileArg, QuantTileArgLaunch, Space, Storage, TileArg, TileArgLaunch,
};

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
/// transparently dequantize on read, i.e. produce `dequant(q) + scalar = q * scale + scalar`.
/// Uses real native (unpacked) Q8S quantization; packing isn't implemented yet. `scalar_add`
/// currently reads the raw quantized value and ignores the scale, so this FAILS until the
/// quant-aware tile read applies it.
/// NOTE: native Q8S stores i8, which wgpu can't compile — run on a CPU/CUDA backend.
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
        QuantTileArgLaunch::new(
            input.tensor_arg(1),
            ComptimeOptionArgs::None,
            input.space(),
            input.storage(),
        ),
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
    input: &QuantTileArg<'_, I, N>,
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

/// Launch `scalar_add` over a native (unpacked) Q8S quantized input and check
/// `out == q * scale + scalar`.
fn run_quantized(m: usize, n: usize, scalar: f32) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let scale = 0.05f32;

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);

    let shape = Shape::from(vec![m, n]);
    let input_dtype = StorageType::Scalar(ElemType::from_quant_value(scheme.value));
    let (lo, hi) = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();

    let space = Space::new(&[(M, m), (N, n)]);
    let storage = Storage::of(2, 2);
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    // Per-tensor scale grid: a single [1] f32 tensor carried alongside the quantized input.
    let scales = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![scale])
        .generate_without_host_data();

    let out_dtype = f32::as_type_native_unchecked().storage_type();
    scalar_add::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        QuantTileArgLaunch::new(
            input.binding().into_tensor_arg(),
            ComptimeOptionArgs::Some(QuantArgLaunch::new(
                scales.binding().into_tensor_arg(),
                scheme,
            )),
            space,
            storage,
        ),
        scalar,
        TileArgLaunch::new(output.tensor_arg(1), output.space(), output.storage()),
        input_dtype,
        out_dtype,
        1usize,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| input_host.get_f32(&idx) * scale + scalar)
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };

    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}
