use cubecl::{
    Runtime, TestRuntime, ir::ElemType, prelude::*, std::tensor::TensorHandle, zspace::Shape,
};
use cubek_quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, StridedLayout, TestInput, assert_equals_approx,
};

const SCALE: f32 = 0.05;
const SEED: u64 = 0x1;

#[test]
fn dequantize_tiled_native_per_tensor_matches_reference() {
    dequantize_tiled_native_per_tensor(&[128, 128]);
}

fn dequantize_tiled_native_per_tensor(tensor_shape: &[usize]) {
    let client = TestRuntime::client(&Default::default());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32);

    let shape = Shape::from(tensor_shape.to_vec());
    let input_dtype = StorageType::Scalar(ElemType::from_quant_value(scheme.value));

    let input_range = scheme.value.range();
    let (input, input_host) = TestInput::builder(client.clone(), shape.clone())
        .dtype(input_dtype)
        .uniform(SEED, input_range.0, input_range.1)
        .generate_with_f32_host_data();

    let scales = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![SCALE])
        .generate_without_host_data();

    let output = TensorHandle::zeros(&client, shape.clone(), f32::as_type_native_unchecked());
    let output_dtype = f32::as_type_native_unchecked().storage_type();

    cubek_quant::dequantize_tiled::launch_ref::<TestRuntime>(
        &client,
        input.binding(),
        output.clone().binding(),
        scales.binding(),
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
