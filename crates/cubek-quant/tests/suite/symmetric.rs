use cubecl::ir::ElemType;
use cubecl::ir::FloatKind;
use cubecl::server::CopyDescriptor;
use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, zspace::shape};
use cubek_quant::scheme::QuantMode;
use cubek_quant::scheme::QuantScheme;
use cubek_quant::scheme::QuantStore;
use cubek_quant::scheme::QuantValue;

#[test]
fn test_quantization_symmetric_tensor() {
    test_quantization_tensor_symmetric(SHAPE_X, SHAPE_Y, VALUE);
}

#[test]
fn test_quantization_symmetric_block() {
    test_quantization_block_symmetric(
        SHAPE_X, SHAPE_Y, VALUE, SHAPE_X, // Shape x as block_size
    );
}

fn test_quantization_tensor_symmetric(m: usize, n: usize, value: QuantValue) {
    let mode = QuantMode::Symmetric;
    let client = TestRuntime::client(&Default::default());
    let shape = shape![m, n];

    let num_elems: usize = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<_> = (0..num_elems).map(|v| v as f32 - half).collect();
    let input_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data), shape.clone(), f32::type_size());

    let (q_min, q_max) = value.range();
    // input data range is not affected by quant range symmetry
    let scale_f32 = (2.0 * half) / (q_max - q_min);
    let data_scale = vec![scale_f32];

    let scale_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data_scale), shape![1], f32::type_size());

    let input = TensorHandle::new(
        input_alloc.memory,
        shape.clone(),
        input_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let scale = TensorHandle::new(
        scale_alloc.memory,
        shape![1],
        scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let output_f = TensorHandle::zeros(&client, shape, f32::as_type_native_unchecked());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::Tensor)
        .with_mode(mode)
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric);

    // The shape is from the POV of packed u32s.
    let shape_out = shape![m, n / scheme.num_quants()];

    let [output_alloc, output_scale_alloc] = client
        .empty_tensors(vec![
            cubecl::server::MemoryLayoutDescriptor {
                strategy: cubecl::server::MemoryLayoutStrategy::Contiguous,
                shape: shape_out.clone(),
                elem_size: u32::type_size(),
            },
            cubecl::server::MemoryLayoutDescriptor {
                strategy: cubecl::server::MemoryLayoutStrategy::Contiguous,
                shape: shape![1],
                elem_size: f32::type_size(),
            },
        ])
        .try_into()
        .unwrap();
    let output = TensorHandle::new(
        output_alloc.memory,
        shape_out,
        output_alloc.strides,
        u32::as_type_native_unchecked(),
    );
    let output_scale = TensorHandle::new(
        output_scale_alloc.memory,
        shape![1],
        output_scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );

    cubek_quant::quantize::launch_ref(
        &client,
        input.binding(),
        output.clone().binding(),
        scale.binding(),
        output_scale.clone().binding(),
        &scheme,
        ElemType::Float(FloatKind::Flex32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &client,
        // The input of the dequantize kernel is the output of the quantized one.
        output.binding(),
        // We use a new buffer to make sure all values are correctly dequantized back.
        output_f.clone().binding(),
        output_scale.clone().binding(),
        &scheme,
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    let computed = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_f.handle.clone().binding(),
        output_f.shape().clone(),
        output_f.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    let data_restored = f32::from_bytes(&computed);

    // Max quantization error = step size / 2
    let rel_tol = 1e-4;
    let max_error = (scale_f32 / 2.0) * (1f32 + rel_tol);
    assert_eq!(data_restored.len(), data.len());
    for (actual, expected) in data_restored.iter().zip(data.into_iter()) {
        let diff = f32::abs(actual - expected);
        assert!(
            diff <= max_error,
            "Expected: {expected} | Actual: {actual} (diff {diff} > {max_error})"
        );
    }
}

fn test_quantization_block_symmetric(m: usize, n: usize, value: QuantValue, block_size: usize) {
    let mode = QuantMode::Symmetric;
    let client = TestRuntime::client(&Default::default());
    let shape = shape![m, n];

    let num_elems: usize = m * n;
    let half = num_elems as f32 / 2.0;
    let data: Vec<_> = (0..num_elems)
        .map(|v| (v as f32 - half) / num_elems as f32)
        .collect();
    let input_alloc =
        client.create_tensor_from_slice(f32::as_bytes(&data), shape.clone(), f32::type_size());

    let (q_min, q_max) = value.range();

    let scale_count = data.len() / block_size;
    let shape_scale = shape![m, n / block_size];

    let mut scales = Vec::with_capacity(scale_count);

    for block in 0..scale_count {
        let mut c_max = f32::MIN;
        let mut c_min = f32::MAX;

        let block_offset = block * block_size;

        for i in 0..block_size {
            let current = data[block_offset + i];
            c_max = f32::max(c_max, current);
            c_min = f32::min(c_min, current);
        }

        // The bias is assumed to be zero.
        let range = 2.0 * c_min.abs().max(c_max.abs());
        let scale = range / (q_max - q_min);
        scales.push(scale);
    }

    let scale_alloc = client.create_tensor_from_slice(
        f32::as_bytes(&scales),
        shape_scale.clone(),
        f32::type_size(),
    );

    let input = TensorHandle::new(
        input_alloc.memory,
        shape.clone(),
        input_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let scale = TensorHandle::new(
        scale_alloc.memory,
        shape_scale.clone(),
        scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let output_f = TensorHandle::zeros(&client, shape, f32::as_type_native_unchecked());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block([block_size as u8]))
        .with_mode(mode)
        .with_value(value)
        .with_store(QuantStore::PackedU32(0))
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric);

    // The shape is from the POV of packed u32s.
    let shape_out = shape![m, n / scheme.num_quants()];

    let [output_alloc, output_scale_alloc] = client
        .empty_tensors(vec![
            cubecl::server::MemoryLayoutDescriptor {
                strategy: cubecl::server::MemoryLayoutStrategy::Contiguous,
                shape: shape_out.clone(),
                elem_size: u32::type_size(),
            },
            cubecl::server::MemoryLayoutDescriptor {
                strategy: cubecl::server::MemoryLayoutStrategy::Contiguous,
                shape: shape_scale.clone(),
                elem_size: f32::type_size(),
            },
        ])
        .try_into()
        .unwrap();
    let output = TensorHandle::new(
        output_alloc.memory,
        shape_out,
        output_alloc.strides,
        u32::as_type_native_unchecked(),
    );
    let output_scale = TensorHandle::new(
        output_scale_alloc.memory,
        shape_scale.clone(),
        output_scale_alloc.strides,
        f32::as_type_native_unchecked(),
    );

    cubek_quant::quantize::launch_ref(
        &client,
        input.binding(),
        output.clone().binding(),
        scale.binding(),
        output_scale.clone().binding(),
        &scheme,
        ElemType::Float(FloatKind::Flex32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &client,
        // The input of the dequantize kernel is the output of the quantized one.
        output.binding(),
        // We use a new buffer to make sure all values are correctly dequantized back.
        output_f.clone().binding(),
        output_scale.binding(),
        &scheme,
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    let computed = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_f.handle.clone().binding(),
        output_f.shape().clone(),
        output_f.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    let data_restored = f32::from_bytes(&computed);

    assert_eq!(data_restored.len(), data.len());
    let rel_tol = 1e-4;
    for (i, (actual, expected)) in data_restored.iter().zip(data.into_iter()).enumerate() {
        let block = i / block_size;
        let scale = scales[block];
        // Max quantization error = step size / 2
        let max_error = (scale / 2.0) * (1f32 + rel_tol);
        let diff = f32::abs(actual - expected);
        assert!(
            diff <= max_error,
            "Mismatch at {i}, Expected: {expected} | Actual: {actual} (diff {diff} > {max_error})"
        );
    }
}
