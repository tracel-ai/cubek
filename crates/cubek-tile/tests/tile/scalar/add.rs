use cubecl::{TestRuntime, features::TypeUsage, ir::ElemType, prelude::*, zspace::Shape};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TestInput, TileInput, assert_equals_approx,
};
use cubek_tile::{
    Axis, Cut, Leaf, Schedule, Space, Storage, StridedTileArg, StridedTileArgLaunch, Tiling,
    WalkOrder,
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

#[test]
fn scalar_add_quantized_matches_reference() {
    run_quantized(8, 8, -3.0);
}

/// Block-quantized: each `bm×bn` block carries its own scale. Validates that the tile's
/// scale windows in lockstep with the values, so a leaf reads its block's scale.
#[test]
fn scalar_add_quantized_block_matches_reference() {
    run_quantized_block(8, 8, 4, 4, -3.0); // square 2×2 grid of blocks
    run_quantized_block(8, 8, 8, 4, 1.5); // blocks along N only (per-column-group)
    run_quantized_block(8, 8, 4, 8, 1.5); // blocks along M only (per-row-group)
    run_quantized_block(16, 8, 4, 4, -0.5); // non-square tensor, 4×2 grid
    run_quantized_block(6, 8, 4, 4, -0.5); // M's last block is half-filled: the overhang is masked
}

#[cube(launch)]
/// input: the input tensor, storage-typed (`I`); quantized when its payload carries scales
/// scalar: the scalar to add
/// output: the output tensor
pub fn scalar_add<I: Numeric, O: Numeric>(
    input: &StridedTileArg<'_, I>,
    scalar: f32,
    output: &StridedTileArg<'_, O>,
    #[define(I)] _input_dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
) {
    let input = input.tile_dequant::<O>();
    let mut output = output.tile();
    output.add_scalar::<I, f32>(&input, scalar);
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
        StridedTileArgLaunch::strided(input.tensor_arg(1), 1, input.space(), input.storage()),
        scalar,
        StridedTileArgLaunch::strided(output.tensor_arg(1), 1, output.space(), output.storage()),
        dtype,
        dtype,
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

/// Launch `scalar_add` over a native (unpacked) Q8S quantized input and check
/// `out == q * scale + scalar`.
fn run_quantized(m: usize, n: usize, scalar: f32) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return; // backend has no native i8 (e.g. wgpu); native dequant can't run here
    }
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
        StridedTileArgLaunch::strided(input.binding().into_tensor_arg(), 1, space, storage)
            .quantized(scales.binding().into_tensor_arg(), scheme),
        scalar,
        StridedTileArgLaunch::strided(output.tensor_arg(1), 1, output.space(), output.storage()),
        input_dtype,
        out_dtype,
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

/// Launch `scalar_add` over a `bm×bn` block-scaled Q8S input and check each element uses its
/// own block's scale: `out == q * scale[i/bm, j/bn] + scalar`. The space tiles into block-sized
/// leaves so each leaf reads exactly one scale. A tensor that doesn't fill its last block
/// overhangs, and the checked path masks values and scales alike.
fn run_quantized_block(m: usize, n: usize, bm: usize, bn: usize, scalar: f32) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return; // backend has no native i8 (e.g. wgpu); native dequant can't run here
    }

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([bm as u8, bn as u8]))
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

    // A space that tiles into `bm×bn` blocks, one cube walking them: each leaf is one block.
    let space = Tiling::new()
        .extents(&[(M, m), (N, n)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axis(M, Cut::sequential(bm)).axis(N, Cut::sequential(bn))
        })
        .leaf(Leaf::Register);
    // A partial last block overhangs its tile, so reads/writes past the tensor must be masked.
    let check = !m.is_multiple_of(bm) || !n.is_multiple_of(bn);
    let storage = Storage::of(2, space.rank()).checked(check);
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    // One distinct scale per block, row-major over the block grid; a partial block still has one.
    let (sm, sn) = (m.div_ceil(bm), n.div_ceil(bn));
    let scale_vals: Vec<f32> = (0..sm * sn).map(|k| 0.05 * (k + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![sm, sn]))
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let out_dtype = f32::as_type_native_unchecked().storage_type();
    scalar_add::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        StridedTileArgLaunch::strided(input.binding().into_tensor_arg(), 1, space, storage)
            .quantized(scales.binding().into_tensor_arg(), scheme),
        scalar,
        StridedTileArgLaunch::strided(
            output.tensor_arg(1),
            1,
            output.space(),
            output.storage().checked(check),
        ),
        input_dtype,
        out_dtype,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| {
                    let scale = scale_vals[(idx[0] / bm) * sn + (idx[1] / bn)];
                    input_host.get_f32(&idx) * scale + scalar
                })
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };

    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}
