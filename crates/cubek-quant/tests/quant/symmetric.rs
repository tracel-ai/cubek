use cubecl::{
    features::TypeUsage,
    ir::ElemType,
    ir::FloatKind,
    server::CopyDescriptor,
    std::tensor::TensorHandle,
    {TestRuntime, zspace::shape},
};
use cubek_quant::{scheme::QuantMode, scheme::QuantScheme, scheme::QuantStore, scheme::QuantValue};

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
        None,
        output_scale.clone().binding(),
        None,
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &client,
        // The input of the dequantize kernel is the output of the quantized one.
        output.binding(),
        // We use a new buffer to make sure all values are correctly dequantized back.
        output_f.clone().binding(),
        output_scale.clone().binding(),
        None,
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
    for (actual, expected) in data_restored.iter().zip(data) {
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
        None,
        output_scale.clone().binding(),
        None,
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &client,
        // The input of the dequantize kernel is the output of the quantized one.
        output.binding(),
        // We use a new buffer to make sure all values are correctly dequantized back.
        output_f.clone().binding(),
        output_scale.binding(),
        None,
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
    for (i, (actual, expected)) in data_restored.iter().zip(data).enumerate() {
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

#[test]
fn test_quantization_symmetric_block_tensor() {
    // Native storage holds byte-wide values only; the packed sibling covers every value.
    if !matches!(VALUE, QuantValue::Q8F | QuantValue::Q8S) {
        return;
    }

    test_quantization_block_tensor_symmetric(SHAPE_X, SHAPE_Y, VALUE, SHAPE_X, QuantStore::Native);
}

#[test]
fn test_quantization_symmetric_block_tensor_packed() {
    test_quantization_block_tensor_symmetric(
        SHAPE_X,
        SHAPE_Y,
        VALUE,
        SHAPE_X,
        QuantStore::PackedU32(0),
    );
}

/// Two-level: per-block scales normalized by one per-tensor scale.
///
/// The block scales are deliberately split so that neither level alone reconstructs the data. If
/// the kernel dropped the per-tensor scale, or applied it twice, every value would come back off
/// by that factor and the tolerance below would not save it.
fn test_quantization_block_tensor_symmetric(
    m: usize,
    n: usize,
    value: QuantValue,
    block_size: usize,
    store: QuantStore,
) {
    let client = TestRuntime::client(&Default::default());
    if store == QuantStore::Native && !i8::supported_uses(&client).contains(TypeUsage::Conversion) {
        return; // backend has no native i8 (e.g. wgpu), which packed storage does not need
    }

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

    // Raw per-block scales, as a one-level scheme would use.
    let mut raw = Vec::with_capacity(scale_count);
    for block in 0..scale_count {
        let mut amax = 0.0f32;
        for i in 0..block_size {
            amax = amax.max(data[block * block_size + i].abs());
        }
        raw.push(2.0 * amax / (q_max - q_min));
    }

    // Split them: the per-tensor scale takes the magnitude, the block scales keep the spread.
    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    let global_f32 = peak / 4.0;
    let scales: Vec<f32> = raw.iter().map(|s| s / global_f32).collect();

    let scale_alloc = client.create_tensor_from_slice(
        f32::as_bytes(&scales),
        shape_scale.clone(),
        f32::type_size(),
    );
    let global_alloc = client.create_tensor_from_slice(
        f32::as_bytes(&[global_f32]),
        shape![1],
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
    let global = TensorHandle::new(
        global_alloc.memory,
        shape![1],
        global_alloc.strides,
        f32::as_type_native_unchecked(),
    );
    let output_f = TensorHandle::zeros(&client, shape.clone(), f32::as_type_native_unchecked());

    let scheme = QuantScheme::default()
        .with_level(QuantLevel::block_tensor(
            [block_size as u8],
            QuantParam::F32,
        ))
        .with_value(value)
        .with_store(store)
        .with_param(QuantParam::F32)
        .with_mode(QuantMode::Symmetric);

    let output = match store {
        QuantStore::Native => {
            TensorHandle::zeros(&client, shape.clone(), i8::as_type_native_unchecked())
        }
        // The shape is from the POV of packed u32s.
        _ => TensorHandle::zeros(
            &client,
            shape![m, n / scheme.num_quants()],
            u32::as_type_native_unchecked(),
        ),
    };
    let output_scale =
        TensorHandle::zeros(&client, shape_scale.clone(), f32::as_type_native_unchecked());
    let output_global = TensorHandle::zeros(&client, shape![1], f32::as_type_native_unchecked());

    cubek_quant::quantize::launch_ref(
        &client,
        input.binding(),
        output.clone().binding(),
        scale.binding(),
        Some(global.clone().binding()),
        output_scale.clone().binding(),
        Some(output_global.clone().binding()),
        &scheme,
        ElemType::Float(FloatKind::F32),
    )
    .unwrap();

    cubek_quant::dequantize::launch_ref(
        &client,
        output.binding(),
        output_f.clone().binding(),
        output_scale.binding(),
        Some(output_global.clone().binding()),
        &scheme,
        f32::as_type_native_unchecked().storage_type(),
    )
    .unwrap();

    // The per-tensor scale has to survive the round trip: the kernel writes it into the quantized
    // tensor's own region, and dequantize reads it back from there rather than from the input.
    let written = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_global.handle.clone().binding(),
        output_global.shape().clone(),
        output_global.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    assert_eq!(f32::from_bytes(&written)[0], global_f32);

    let computed = client.read_one_unchecked_tensor(CopyDescriptor::new(
        output_f.handle.clone().binding(),
        output_f.shape().clone(),
        output_f.strides().clone(),
        core::mem::size_of::<f32>(),
    ));
    let data_restored = f32::from_bytes(&computed);

    assert_eq!(data_restored.len(), data.len());
    for (block, chunk) in data.chunks(block_size).enumerate() {
        // Error is bounded by half a step of the effective scale, which is the product of the two.
        let max_error = (global_f32 * scales[block] / 2.0) * (1f32 + 1e-4);
        for (i, expected) in chunk.iter().enumerate() {
            let actual = data_restored[block * block_size + i];
            let diff = f32::abs(actual - expected);
            assert!(
                diff <= max_error,
                "block {block} index {i}: got {actual}, expected {expected}, \
                 diff {diff} over {max_error}"
            );
        }
    }
}
