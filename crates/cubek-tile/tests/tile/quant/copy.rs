use cubecl::{
    TestRuntime, features::TypeUsage, ir::ElemType, prelude::*,
    std::tensor::layout::linear::linear_view, zspace::Shape,
};
use cubek_quant::scheme::{QuantMode, QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, MEMORY_LEAF, StridedLayout, TestInput, TestOutcome,
    TileInput, ValidationResult, assert_equals_approx,
};
use cubek_tile::{
    Axis, Buffering, CubeAxis, Cut, DequantAt, QuantTileArg, QuantTileArgLaunch, Space, TileArg,
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

    let input = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

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

/// The op walks the levels that hand regions to different cubes and stops at the plane level,
/// which the transport spreads over the cube itself. Stepping the plane level would leave every
/// plane but the first unwritten, since the fill indexes by the flat unit.
#[test]
fn copy_spread_across_cubes_and_planes_matches_reference() {
    let (m, n) = (4, 512);
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = Tiling::new()
        .extents(&[(M, m), (N, n)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::Y, 1))
                .axis(N, Cut::cube(CubeAxis::X, 128))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(1)).axis(N, Cut::plane(32))
        })
        .build()
        .launcher_over(&client, &[]);
    let space = launch.space().clone();

    let input = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    plain_copy::launch::<TestRuntime>(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        input.arg(),
        output.arg(),
        space,
        f32::elem_type_native(),
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
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let shape = Shape::from(vec![m, n]);
    let input_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();

    let space = Space::new(&[(M, m), (N, n)]);
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![scale])
        .generate_without_host_data();

    let out_dtype = f32::elem_type_native();
    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
        1,
        QuantTileArgLaunch::new(
            input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, N], MEMORY_LEAF),
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

/// Per-tensor native Q8S served in 4-wide lines, through the builder: one full block covers
/// every line, so nothing can straddle a scale and the vectorized operand is accepted.
#[test]
fn copy_quantized_per_tensor_vectorized_matches_reference() {
    let (m, n, v) = (8, 8, 4);
    let scale = 0.05f32;
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }
    let max_width = client.properties().hardware.max_vector_size;
    if v > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below the {v}-wide served line"
        )))
        .enforce();
        return;
    }

    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let shape = Shape::from(vec![m, n]);
    let input_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![scale])
        .generate_without_host_data();

    let space = Space::new(&[(M, m), (N, n)]);
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    let launcher = space.launcher_over(&client, &[]);
    let input_op = launcher
        .arg(input.binding(), MEMORY_LEAF)
        .subspace(&[M, N])
        .vectorize(v)
        .quantized(&[scales.binding()], scheme, DequantAt::Read)
        .build();

    let out_dtype = f32::elem_type_native();
    dequant_copy::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        input_op.bound_width(),
        v,
        input_op.arg(),
        output.arg(),
        launcher.space().clone(),
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

/// Per-tensor packed-u32 Q8S, through the builder: one full block covers whole `pack`-wide
/// lines. The binding is a `u32`, so this runs on every backend, no native i8 needed.
#[test]
fn copy_quantized_per_tensor_packed_matches_reference() {
    let (m, n) = (8, 8);
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

    let space = Space::new(&[(M, m), (N, n)]);
    let input = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    let launcher = space.launcher_over(&client, &[]);
    let input_op = launcher
        .arg(input.tile.handle().binding(), MEMORY_LEAF)
        .subspace(&[M, N])
        .vectorize(pack)
        .quantized(&[input.scales_binding()], scheme, DequantAt::Read)
        .build();

    let input_dtype = u32::elem_type_native();
    let out_dtype = f32::elem_type_native();
    dequant_copy::launch::<TestRuntime>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        input_op.bound_width(),
        pack,
        input_op.arg(),
        output.arg(),
        launcher.space().clone(),
        input_dtype,
        out_dtype,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let shape = Shape::from(vec![m, n]);
    let expected = HostData {
        data: HostDataVec::F32(
            (0..m * n)
                .map(|k| input.q[k] as f32 * input.scale_values[0])
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
    run_quantized_block(8, 8, 4, 4, None); // square 2×2 grid of blocks
    run_quantized_block(8, 8, 8, 4, None); // blocks along N only (per-column-group)
    run_quantized_block(8, 8, 4, 8, None); // blocks along M only (per-row-group)
    run_quantized_block(16, 8, 4, 4, None); // non-square tensor, 4×2 grid
    run_quantized_block(6, 8, 4, 4, None); // M's last block is half-filled: the overhang is masked
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

/// Packed-u32 lookup-quantized ([`QuantMode::Lookup`]): each 4-bit field is an index into a
/// 16-entry table, so `out == table[q] * scale[i/bm, j/bn]`. The table is deliberately not
/// affine in the index — a decode that fell back to the integer cast would reconstruct the
/// index itself and miss every entry. Block scales beside it pin that the two lookups (block →
/// scale, field → entry) stay independent.
#[test]
fn copy_quantized_lookup_matches_reference() {
    let (m, n, bm, bn) = (8usize, 8usize, 4usize, 8usize);
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let scheme = QuantScheme::default()
        .per_block([bm as u8, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q4F)
        .with_mode(QuantMode::Lookup);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below the lookup scheme's packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    // Ascending and centroid-like, nothing an integer cast could reproduce.
    let table: [f32; 16] = [
        -100.0, -10.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 8.0,
        42.0,
    ];

    let space = Space::new(&[(M, m), (N, n)]);
    let input = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .lookup_arange(&table);
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
        pack,
        input.arg(),
        output.arg(),
        space,
        u32::elem_type_native(),
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let sn = n / bn;
    let shape = Shape::from(vec![m, n]);
    let expected = HostData {
        data: HostDataVec::F32(
            (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    input.table_values[input.q[k] as usize]
                        * input.scale_values[(i / bm) * sn + (j / bn)]
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

/// Sub-word packed-u32: the output's line is **narrower than a word**, so the source serves
/// one-line-per-word (a scalar `u32` binding) and the fill unpacks each word across
/// `num_quants / w` lines (`scan_words`). This is the regime a vec4 device reads 4- and 2-bit
/// caches in; it needs no width skip, which is the point. The innermost block covers whole
/// words, `scan_words`' scale rule.
#[test]
fn copy_quantized_subword_matches_reference() {
    run_quantized_subword(8, 8, QuantValue::Q4S, 4, 8, 4); // 8 per word, 2 lines each
    run_quantized_subword(8, 16, QuantValue::Q2S, 8, 16, 4); // 16 per word, 4 lines each
    run_quantized_subword(8, 8, QuantValue::Q8S, 4, 4, 2); // 4 per word, 2 lines each
}

/// [`run_quantized_packed`]'s sub-word twin: the input serves whole words, the output is
/// vectorized at `w < num_quants`.
fn run_quantized_subword(m: usize, n: usize, value: QuantValue, bm: usize, bn: usize, w: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let scheme = QuantScheme::default()
        .per_block([bm as u8, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
    let pack = scheme.num_quants();
    assert!(w < pack && pack.is_multiple_of(w));

    let space = Space::new(&[(M, m), (N, n)]);
    let input = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .packed(&scheme, DequantAt::Load)
        .arange();
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
        w,
        input.arg(),
        output.arg(),
        space,
        u32::elem_type_native(),
        f32::elem_type_native(),
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

/// Sub-word **lookup**: each word's fields index the table and unpack across several output
/// lines — the exact read a vec4 device gives a 4-bit lookup-quantized cache.
#[test]
fn copy_quantized_subword_lookup_matches_reference() {
    let (m, n, bm, bn, w) = (8usize, 8usize, 4usize, 8usize, 4usize);
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let scheme = QuantScheme::default()
        .per_block([bm as u8, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q4F)
        .with_mode(QuantMode::Lookup);

    let table: [f32; 16] = [
        -100.0, -10.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 2.0, 8.0,
        42.0,
    ];

    let space = Space::new(&[(M, m), (N, n)]);
    let input = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .packed(&scheme, DequantAt::Load)
        .lookup_arange(&table);
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
        w,
        input.arg(),
        output.arg(),
        space,
        u32::elem_type_native(),
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let sn = n / bn;
    let shape = Shape::from(vec![m, n]);
    let expected = HostData {
        data: HostDataVec::F32(
            (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    input.table_values[input.q[k] as usize]
                        * input.scale_values[(i / bm) * sn + (j / bn)]
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

/// Copy a `bm×bn` block-scaled packed input and check each value used its own block's scale:
/// `out == q * scale[i/bm, j/bn]`.
///
/// The packed operand is described **in values** — shape `[m, n]`, strides `[n, 1]` — while its
/// buffer holds `m·n/pack` `u32`s. That is the launch convention the served-width split rests on:
/// the tile counts lines, and one `u32` line is one served line of `pack` values.
fn run_quantized_packed(m: usize, n: usize, value: QuantValue, bm: usize, bn: usize) {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let scheme = QuantScheme::default()
        .per_block([bm as u8, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
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
    let input = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    let input_dtype = u32::elem_type_native();
    let out_dtype = f32::elem_type_native();
    // The packed binding stays a scalar `u32`: the scheme serves `pack` values per word,
    // so the copy moves whole lines and the destination is lined at that served width.
    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
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
/// quantized input dequantizes with nothing threaded through the body. `VI` is the binding
/// width (served ÷ packing factor), `VO` the served width the copy writes at.
pub fn dequant_copy<I: Numeric, O: Numeric, VI: Size, VO: Size>(
    input: &QuantTileArg<'_, I, VI>,
    output: &TileArg<'_, O, VO>,
    #[comptime] space: Space,
    #[define(I)] _input_dtype: ElemType,
    #[define(O)] _output_dtype: ElemType,
) {
    let input = input.tile::<O>(comptime!(space.clone()));
    let mut output = output.tile(space);
    output.copy_from(&input);
}

/// Two-level: block scales normalized by one per-tensor scale, both folded into every read,
/// `out == q * scale[i/bm, j/bn] * global`. The expectation carries the global scale, so an
/// implementation that drops it fails by exactly that factor.
#[test]
fn copy_quantized_two_level_matches_reference() {
    run_quantized_block(8, 8, 4, 4, Some(0.5));
    run_quantized_block(16, 8, 4, 4, Some(0.25));
    run_quantized_block(6, 8, 4, 4, Some(0.5)); // M's last block is half-filled: masked overhang
    // The whole window fits inside one block: `QuantInfo::uniform()` holds, so this exercises
    // `uniform_scale()`'s whole-scale fold instead of the per-position one under `KnownScale::Global`.
    run_quantized_block(4, 4, 4, 4, Some(0.5));
}

/// The mutation check on the same path: a zero per-tensor scale zeroes every reconstruction, so
/// the global scale provably participates in each read rather than defaulting to one.
#[test]
fn copy_quantized_two_level_zero_global_scale_zeroes_output() {
    run_quantized_block(8, 8, 4, 4, Some(0.0));
}

/// A two-level scheme with no global binding is refused by the builder, host-side and on the
/// caller's thread: a missing per-tensor scale would otherwise reconstruct every value short by
/// that factor. (The kernel-side backstop in `QuantTileArg::tile` cannot be pinned here: it fires
/// on the compile server, where a panic is swallowed rather than propagated.)
#[test]
#[should_panic(expected = "takes as many scale bindings")]
fn two_level_without_global_scale_refused_by_the_builder() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let (m, n) = (8, 8);
    let scheme = two_level_scheme(4, 4);
    let shape = Shape::from(vec![m, n]);
    let input_dtype = ElemType::from_quant_value(scheme.value);
    let input = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(0x1, -8.0, 7.0)
        .generate_without_host_data();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![2usize, 2]))
        .custom(vec![1.0; 4])
        .generate_without_host_data();

    let space = Space::new(&[(M, m), (N, n)]);
    let launcher = space.launcher(&client);
    launcher
        .arg(input.binding(), MEMORY_LEAF)
        .subspace(&[M, N])
        .quantized(&[scales.binding()], scheme, DequantAt::Read)
        .build();
}

fn two_level_scheme(bm: usize, bn: usize) -> QuantScheme {
    QuantScheme::default()
        .per_block([bm as u8, bn as u8], ScaleDtype::F32)
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
}

/// Copy a `bm×bn` block-scaled Q8S input and check each element used its own block's scale:
/// `out == q * scale[i/bm, j/bn]`, or with `global` set (two-level), `out == q * scale[..] *
/// global`, the global scale bound as a third 1-element tensor. The space tiles into block-sized
/// leaves, so a tensor that doesn't fill its last block overhangs it.
fn run_quantized_block(m: usize, n: usize, bm: usize, bn: usize, global: Option<f32>) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
        return;
    }

    let block_scheme = QuantScheme::default()
        .per_block([bm as u8, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scheme = match global {
        Some(_) => block_scheme.per_tensor(ScaleDtype::F32),
        None => block_scheme,
    };

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
    let output = TileInput::builder(&client, space.clone(), MEMORY_LEAF)
        .untiled()
        .zeros();

    // One distinct scale per block, row-major over the block grid; a partial block still has one.
    let (sm, sn) = (m.div_ceil(bm), n.div_ceil(bn));
    let scale_vals: Vec<f32> = (0..sm * sn).map(|k| 0.05 * (k + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![sm, sn]))
        .custom(scale_vals.clone())
        .generate_without_host_data();
    let global_scale = global.map(|g| {
        TestInput::builder(client.clone(), Shape::from(vec![1usize]))
            .custom(vec![g])
            .generate_without_host_data()
    });

    let out_dtype = f32::elem_type_native();
    dequant_copy::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        1,
        1,
        QuantTileArgLaunch::new(
            input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            global_scale.map(|g| linear_view(g.binding())).into(),
            None.into(),
            TileSpec::direct(&[M, N], MEMORY_LEAF),
            scheme,
            DequantAt::Read,
        ),
        TileArgLaunch::new(output.tensor_arg(1), output.spec().checked(check)),
        space,
        input_dtype,
        out_dtype,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let g = global.unwrap_or(1.0);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| {
                    let scale = scale_vals[(idx[0] / bm) * sn + (idx[1] / bn)];
                    input_host.get_f32(&idx) * scale * g
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
