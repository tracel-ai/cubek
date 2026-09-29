use cubecl::{ir::ElemType, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubek_quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
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
    let client = cubecl::test_device().client();

    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let shape = Shape::from(tensor_shape.to_vec());
    let input_dtype = ElemType::from_quant_value(scheme.value);

    // The values as the store keeps them, one byte each, written from the host: a device without
    // native `i8` cannot generate them, and the decode under test never needs it.
    let (lo, hi) = scheme.value.range();
    let span = (hi - lo) as i64 + 1;
    let values: Vec<i8> = (0..shape.num_elements() as u64)
        .map(|i| {
            (lo as i64 + ((i.wrapping_mul(0x9e37_79b9).wrapping_add(SEED)) as i64).rem_euclid(span))
                as i8
        })
        .collect();
    let input = TensorHandle::new_contiguous(
        tensor_shape.to_vec(),
        client.create(cubecl::bytes::Bytes::from_elems(values.clone())),
        input_dtype,
    );
    let input_host = HostData {
        data: HostDataVec::F32(values.iter().map(|&v| v as f32).collect()),
        strides: StridedLayout::RowMajor.compute_strides(&shape),
        shape: shape.clone(),
    };

    let scales = TestInput::builder(client.clone(), Shape::from(vec![1usize]))
        .custom(vec![SCALE])
        .generate_without_host_data();

    let output = TensorHandle::zeros(&client, shape.clone(), f32::elem_type_native());
    let output_dtype = f32::elem_type_native();

    cubek_quant::dequantize_tiled::launch_ref(
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
