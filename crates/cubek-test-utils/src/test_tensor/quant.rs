use cubecl::{
    TestRuntime, client::ComputeClient, ir::ElemType, std::tensor::TensorHandle, zspace::Shape,
};
use cubecl_common::quant::scheme::QuantScheme;

use crate::{
    HostData, QuantizationInfo, TestTensor, quant_layout, stubs::quant::quantize,
    test_tensor::custom::cast_f32_to_dtype,
};

/// Quantize an already-built [`TestTensor`] in place: runs the host-side
/// reference quantizer ([`crate::stubs::quant`]) over the tensor's host data,
/// swaps the handle for the packed output, and stores the scale + original
/// shape on `tensor.quantization`. The host data on `tensor.host` is left as
/// the original f32 reference so correctness checks can still compare against it.
pub(crate) fn apply_quantization(
    client: &ComputeClient<TestRuntime>,
    tensor: &mut TestTensor,
    scheme: QuantScheme,
) {
    let original_shape = tensor.handle.shape().clone();

    // Derive the scale tensor from the quant value's range. For
    // a per-tensor scheme a single scale is used; for a block scheme we
    // compute a per-block scale from the host data so each block fully uses
    // the quant range without clipping.
    let (scales_shape, scales_data, block_dims) = compute_input_scales(&tensor.host, &scheme);

    // Quantize on the host (see `crate::stubs::quant`). The kernel's `out_scale`
    // is simply the input scales cast to the param precision, so we build it
    // directly from `scales_data` instead of having the stub recompute it.
    let shape: Vec<usize> = original_shape.iter().copied().collect();
    let output_storage_type = quant_layout::values_dtype(&scheme);
    let quant_shape = Shape::from(quant_layout::values_shape(&scheme, &shape));
    let values = logical_values_f32(&tensor.host);
    let output_bytes = quantize(&values, &shape, &scales_data, &block_dims, &scheme);
    let output_handle = TensorHandle::new_contiguous(
        quant_shape,
        client.create_from_slice(&output_bytes),
        output_storage_type,
    );

    let scale_dtype = ElemType::from_scale_dtype(scheme.scale_dtype());
    let out_scale_bytes = cast_f32_to_dtype(&scales_data, scale_dtype);
    let out_scale_handle = TensorHandle::new_contiguous(
        scales_shape,
        client.create_from_slice(&out_scale_bytes),
        scale_dtype,
    );

    // Keep the packed shape on the handle.
    tensor.handle = output_handle;
    tensor.quantization = Some(QuantizationInfo {
        scheme,
        scale: out_scale_handle,
        shape: original_shape,
    });
}

/// Flatten the host data into a logical, row-major `Vec<f32>` for the
/// host-side quantizer.
fn logical_values_f32(host: &HostData) -> Vec<f32> {
    let shape: Vec<usize> = host.shape.iter().copied().collect();
    let rank = shape.len();
    let num_elems: usize = shape.iter().product();

    let mut values = Vec::with_capacity(num_elems);
    let mut idx = vec![0usize; rank];
    for linear in 0..num_elems {
        let mut rem = linear;
        for d in (0..rank).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        values.push(host.get_f32(&idx));
    }
    values
}

/// Compute the scale tensor shape, values, and per-dimension block extent for a
/// quantized input.
///
/// For a per-tensor scheme this returns a single-element scale based on the
/// quant value's range (assumes input in [-1, 1]) and the full shape as the
/// block. For a block scheme each block gets its own scale derived from
/// `max(|value|)` in that block, matching the reference pattern used by the
/// cubek-quant symmetric tests.
fn compute_input_scales(host: &HostData, scheme: &QuantScheme) -> (Shape, Vec<f32>, Vec<usize>) {
    let (q_min, q_max) = scheme.value.range();
    let max_abs_q = q_max.abs().max(q_min.abs());

    let shape: Vec<usize> = host.shape.iter().copied().collect();
    let block_dims = crate::stubs::quant::block_dims(scheme, &shape);
    let scales_shape: Shape = crate::quant_layout::scales_grid(&shape, &block_dims)
        .into_iter()
        .collect();

    let scales = if scheme.block_size().is_none() {
        vec![1.0 / max_abs_q]
    } else {
        let rank = shape.len();
        let num_blocks: usize = scales_shape.iter().product();
        let block_elem_count: usize = block_dims.iter().product();

        let mut scales = Vec::with_capacity(num_blocks);
        let mut data_idx = vec![0usize; rank];
        for block_linear in 0..num_blocks {
            // Decode the flat block index into per-dim block indices.
            let mut block_idx = vec![0usize; rank];
            let mut rem = block_linear;
            for d in (0..rank).rev() {
                block_idx[d] = rem % scales_shape[d];
                rem /= scales_shape[d];
            }

            let mut block_max = 0.0_f32;
            for elem_linear in 0..block_elem_count {
                let mut rem = elem_linear;
                for d in (0..rank).rev() {
                    let within = rem % block_dims[d];
                    data_idx[d] = block_idx[d] * block_dims[d] + within;
                    rem /= block_dims[d];
                }
                block_max = block_max.max(host.get_f32(&data_idx).abs());
            }

            // Guard against an all-zero block producing a zero scale that
            // would divide-by-zero inside the quantize kernel.
            let scale = if block_max > 0.0 {
                block_max / max_abs_q
            } else {
                1.0 / max_abs_q
            };
            scales.push(scale);
        }
        scales
    };

    (scales_shape, scales, block_dims)
}
