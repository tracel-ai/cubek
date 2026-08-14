use cubecl::{TestRuntime, features::TypeUsage, ir::ElemType, prelude::*, zspace::Shape};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TestInput, TestOutcome, TileInput,
    ValidationResult, assert_equals_approx,
};
use cubek_tile::{
    Axis, Buffering, Cut, DequantAt, QuantTileArg, QuantTileArgLaunch, Space, TileArg,
    TileArgLaunch, TileSpec, Tiling, WalkOrder,
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);

/// Base sanity: a plain (non-quantized) tile copies through unchanged.
#[test]
fn copy_non_quantized_matches_reference() {
    let (m, n) = (8, 8);
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let space = Space::new(&[(M, m), (N, n)]);

    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    let dtype = f32::elem_type_native();
    plain_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        input.arg(),
        output.arg(),
        space,
        dtype,
    );

    let input_host = HostData::from_tensor_handle(&client, input.handle(), HostDataType::F32);
    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    assert_equals_approx(&got, &input_host, 1e-6)
        .as_test_outcome()
        .enforce();
}

/// Per-tensor native Q8S through the plain copy: `out == q * scale`, with no `I` in the kernel.
#[test]
fn copy_quantized_per_tensor_matches_reference() {
    let (m, n) = (8, 8);
    let scale = 0.05f32;
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);

    let shape = Shape::from(vec![m, n]);
    let input_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();

    let space = Space::new(&[(M, m), (N, n)]);
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![scale])
        .generate_without_host_data();

    let out_dtype = f32::elem_type_native();
    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
        QuantTileArgLaunch::new(
            input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
            scheme,
            DequantAt::Read,
        ),
        output.arg(),
        space,
        input_dtype,
        out_dtype,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| input_host.get_f32(&idx) * scale)
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };
    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}

/// Block-quantized: each `bm×bn` block carries its own scale, and one flat fill spans the whole
/// grid — the per-line lookup picks each line's scale. The last case's tiles overhang the tensor,
/// running the checked path; only the valid region is asserted, so it pins that masking leaves
/// live values (and their scales) intact, not that the overhang itself is suppressed.
#[test]
fn copy_quantized_block_matches_reference() {
    run_quantized_block(8, 8, 4, 4); // square 2×2 grid of blocks
    run_quantized_block(8, 8, 8, 4); // blocks along N only (per-column-group)
    run_quantized_block(8, 8, 4, 8); // blocks along M only (per-row-group)
    run_quantized_block(16, 8, 4, 4); // non-square tensor, 4×2 grid
    run_quantized_block(6, 8, 4, 4); // M's last block is half-filled: the overhang is masked
}

/// Packed-u32 block-quantized: the buffer holds `num_quants` values per `u32`, which the view
/// unpacks on read. Unlike the native cases this needs no i8 support — the binding is a `u32` —
/// so it runs on every backend.
///
/// Each case's inner block is a multiple of the served line, as the launch requires (a line may
/// not split a `u32`, nor straddle two scales). A whole word is one served line, so a scheme's
/// packing factor must fit the device's vector width — a case that doesn't is skipped loudly,
/// the same gate a selector applies when it picks widths from the device (only WGSL-bound
/// targets cap at 4; cpu/cuda serve any width).
#[test]
fn copy_quantized_packed_u32_matches_reference() {
    // Q8S packs 4 values per u32.
    run_quantized_packed(8, 8, QuantValue::Q8S, 4, 4); // square 2×2 grid of blocks
    run_quantized_packed(8, 8, QuantValue::Q8S, 4, 8); // blocks along M only
    run_quantized_packed(16, 8, QuantValue::Q8S, 4, 4); // non-square tensor
    // Q4S packs 8 values per u32, so a block must cover at least a whole word.
    run_quantized_packed(8, 8, QuantValue::Q4S, 4, 8);
    run_quantized_packed(8, 16, QuantValue::Q4S, 8, 8);
}

/// Copy a `bm×bn` block-scaled packed input and check each value used its own block's scale:
/// `out == q * scale[i/bm, j/bn]`.
///
/// The packed operand is described **in values** — shape `[m, n]`, strides `[n, 1]` — while its
/// buffer holds `m·n/pack` `u32`s. That is the launch convention the served-width split rests on:
/// the tile counts lines, and one `u32` line is one served line of `pack` values.
fn run_quantized_packed(m: usize, n: usize, value: QuantValue, bm: usize, bn: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([bm as u8, bn as u8]))
        .with_store(QuantStore::PackedU32(0))
        .with_value(value)
        .with_param(QuantParam::F32);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below {value:?}'s packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let space = Space::new(&[(M, m), (N, n)]);
    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    let input_dtype = u32::elem_type_native();
    let out_dtype = f32::elem_type_native();
    // The packed binding stays a scalar `u32`: the scheme serves `pack` values per word,
    // so the copy moves whole lines and the destination is lined at that served width.
    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        pack,
        input.arg(),
        output.arg(),
        space,
        input_dtype,
        out_dtype,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let sn = n / bn;
    let shape = Shape::from(vec![m, n]);
    let expected = HostData {
        data: HostDataVec::F32(
            (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    input.q[k] as f32 * input.scale_values[(i / bm) * sn + (j / bn)]
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

#[cube(launch)]
/// A plain (non-quantized) copy: both tiles serve `E` straight from their tensors.
pub fn plain_copy<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(comptime!(space.clone()));
    let mut output = output.tile(space);
    output.copy_from(&input);
}

#[cube(launch)]
/// `I` names the binding's element only: the arg's scheme recovers the served value, so a
/// quantized input dequantizes with nothing threaded through the body.
pub fn dequant_copy<I: Numeric, O: Numeric, V: Size>(
    input: &QuantTileArg<'_, I, Const<1>>,
    output: &TileArg<'_, O, V>,
    #[comptime] space: Space,
    #[define(I)] _input_dtype: ElemType,
    #[define(O)] _output_dtype: ElemType,
) {
    let input = input.tile::<O>(comptime!(space.clone()));
    let mut output = output.tile(space);
    output.copy_from(&input);
}

/// Copy a `bm×bn` block-scaled Q8S input and check each element used its own block's scale:
/// `out == q * scale[i/bm, j/bn]`. The space tiles into block-sized leaves, so a tensor that
/// doesn't fill its last block overhangs it.
fn run_quantized_block(m: usize, n: usize, bm: usize, bn: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([bm as u8, bn as u8]))
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);

    let shape = Shape::from(vec![m, n]);
    let input_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();

    // A space that tiles into `bm×bn` blocks, one cube walking them.
    let space = Tiling::new()
        .extents(&[(M, m), (N, n)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(bm)).axis(N, Cut::sequential(bn))
        })
        .build();
    // A partial last block overhangs its tile, so reads/writes past the tensor must be masked.
    let check = !m.is_multiple_of(bm) || !n.is_multiple_of(bn);
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    // One distinct scale per block, row-major over the block grid; a partial block still has one.
    let (sm, sn) = (m.div_ceil(bm), n.div_ceil(bn));
    let scale_vals: Vec<f32> = (0..sm * sn).map(|k| 0.05 * (k + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![sm, sn]))
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let out_dtype = f32::elem_type_native();
    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
        QuantTileArgLaunch::new(
            input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
            scheme,
            DequantAt::Read,
        ),
        TileArgLaunch::new(output.tensor_arg(1), output.spec().checked(check)),
        space,
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
                    input_host.get_f32(&idx) * scale
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
