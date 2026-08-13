use cubecl::{
    Runtime, TestRuntime, features::TypeUsage, ir::ElemType, prelude::*, std::tensor::TensorHandle,
    zspace::Shape,
};
use cubek_quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TestInput, TileInput, assert_equals_approx,
};
use cubek_tile::{Axis, DequantAt, Space};

const SCALE: f32 = 0.05;
const SEED: u64 = 0x1;

#[test]
fn dequantize_tiled_native_per_tensor_matches_reference() {
    dequantize_tiled_native_per_tensor(&[128, 128]);
}

/// An extent nothing divides serves scalar (bounds-checked where the cube tile overhangs);
/// same numbers as the vectorized launch, only the plan degrades.
#[test]
fn dequantize_tiled_native_per_tensor_awkward_shape_matches_reference() {
    dequantize_tiled_native_per_tensor(&[6, 127]);
}

#[test]
fn dequantize_tiled_native_per_tensor_rank_3_matches_reference() {
    dequantize_tiled_native_per_tensor(&[3, 5, 64]);
}

/// An innermost extent the preferred multi-plane cube divides: the launch raises the cube's
/// unit count through the plane level instead of multiplying cubes.
#[test]
fn dequantize_tiled_native_per_tensor_multi_plane_matches_reference() {
    dequantize_tiled_native_per_tensor(&[4, 4096]);
}

/// A `[32]`-block scheme reads each innermost 32-value group through its own scale.
#[test]
fn dequantize_tiled_native_block_matches_reference() {
    let client = TestRuntime::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return; // backend has no native i8 (e.g. wgpu); native dequant can't run here
    }
    let (m, n, bn) = (64, 128, 32);

    let scheme = QuantScheme::default()
        .per_block([bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let shape = Shape::from(vec![m, n]);
    let input_dtype = ElemType::from_quant_value(scheme.value);
    let input_range = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(SEED, input_range.0, input_range.1)
        .generate_with_f32_host_data();

    let sn = n / bn;
    let scale_vals: Vec<f32> = (0..m * sn).map(|k| 0.05 * (k + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![m, sn]))
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let output = TensorHandle::zeros(&client, shape.clone(), f32::elem_type_native());
    cubek_quant::dequantize_tiled::launch_ref::<TestRuntime>(
        &client,
        input.binding(),
        output.clone().binding(),
        &[scales.binding()],
        &scheme,
        f32::elem_type_native(),
    )
    .unwrap();

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| input_host.get_f32(&idx) * scale_vals[idx[0] * sn + idx[1] / bn])
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };
    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}

/// Packed-u32 Q8S with a `[32]`-block: the binding is a `u32` declared in values, so this
/// runs on every backend, no native i8 needed.
#[test]
fn dequantize_tiled_packed_block_matches_reference() {
    let client = TestRuntime::client(&Default::default());
    let (m, n, bn) = (16, 64, 32);

    let scheme = QuantScheme::default()
        .per_block([bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S);
    let pack = scheme.num_quants();
    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        return; // no served line can cover a whole u32 word here
    }

    let space = Space::new(&[(Axis(0), m), (Axis(1), n)]);
    let input = TileInput::builder(&client, space)
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();

    let sn = n / bn;
    let scales = TestInput::builder(client.clone(), Shape::from(vec![m, sn]))
        .custom(input.scale_values.clone())
        .generate_without_host_data();

    let shape = Shape::from(vec![m, n]);
    let output = TensorHandle::zeros(&client, shape.clone(), f32::elem_type_native());
    cubek_quant::dequantize_tiled::launch_ref::<TestRuntime>(
        &client,
        input.tile.handle().binding(),
        output.clone().binding(),
        &[scales.binding()],
        &scheme,
        f32::elem_type_native(),
    )
    .unwrap();

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            (0..m * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    input.q[k] as f32 * input.scale_values[i * sn + j / bn]
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

/// Two-level: block scales normalized by one per-tensor scale, both folded into every value.
/// The expectation carries the global, so an implementation that drops it fails by that factor.
#[test]
fn dequantize_tiled_native_two_level_matches_reference() {
    dequantize_tiled_native_two_level(0.5);
}

/// A zero per-tensor scale zeroes every reconstruction, so the global provably participates
/// in each value rather than defaulting to one.
#[test]
fn dequantize_tiled_native_two_level_zero_global_zeroes_output() {
    dequantize_tiled_native_two_level(0.0);
}

fn dequantize_tiled_native_per_tensor(tensor_shape: &[usize]) {
    let client = TestRuntime::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return; // backend has no native i8 (e.g. wgpu); native dequant can't run here
    }

    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let shape = Shape::from(tensor_shape.to_vec());
    let input_dtype = ElemType::from_quant_value(scheme.value);

    let input_range = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(SEED, input_range.0, input_range.1)
        .generate_with_f32_host_data();

    let scales = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![SCALE])
        .generate_without_host_data();

    let output = TensorHandle::zeros(&client, shape.clone(), f32::elem_type_native());
    let output_dtype = f32::elem_type_native();

    cubek_quant::dequantize_tiled::launch_ref::<TestRuntime>(
        &client,
        input.binding(),
        output.clone().binding(),
        &[scales.binding()],
        &scheme,
        output_dtype,
    )
    .unwrap();

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| input_host.get_f32(&idx) * SCALE)
                .collect(),
        ),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape,
    };
    assert_equals_approx(&got, &expected, 1e-6)
        .as_test_outcome()
        .enforce();
}

fn dequantize_tiled_native_two_level(global: f32) {
    let client = TestRuntime::client(&Default::default());
    if !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return; // backend has no native i8 (e.g. wgpu); native dequant can't run here
    }
    let (m, n, bn) = (32, 64, 16);

    let scheme = QuantScheme::default()
        .per_block([bn as u8], ScaleDtype::F32)
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let shape = Shape::from(vec![m, n]);
    let input_dtype = ElemType::from_quant_value(scheme.value);
    let input_range = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(SEED, input_range.0, input_range.1)
        .generate_with_f32_host_data();

    let sn = n / bn;
    let scale_vals: Vec<f32> = (0..m * sn).map(|k| 0.05 * (k + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), Shape::from(vec![m, sn]))
        .custom(scale_vals.clone())
        .generate_without_host_data();
    let global_scale = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![global])
        .generate_without_host_data();

    let output = TensorHandle::zeros(&client, shape.clone(), f32::elem_type_native());
    cubek_quant::dequantize_tiled::launch_ref::<TestRuntime>(
        &client,
        input.binding(),
        output.clone().binding(),
        &[scales.binding(), global_scale.binding()],
        &scheme,
        f32::elem_type_native(),
    )
    .unwrap();

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    let expected = HostData {
        data: HostDataVec::F32(
            input_host
                .iter_indices()
                .map(|idx| {
                    input_host.get_f32(&idx) * scale_vals[idx[0] * sn + idx[1] / bn] * global
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
