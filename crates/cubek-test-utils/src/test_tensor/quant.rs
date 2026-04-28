use cubecl::{
    TestRuntime,
    client::ComputeClient,
    ir::{ElemType, StorageType},
    zspace::Shape,
};
use cubecl_common::quant::scheme::QuantScheme;
use cubek_quant::scheme::{QuantLevel, QuantStore};

use crate::{HostData, QuantizationInfo, TestInput, TestTensor};

/// Quantize an already-built [`TestTensor`] in place: launches the cubek-quant
/// reference quantizer on `tensor.handle`, swaps the handle for the packed
/// output, and stores the device-side scale + original shape on
/// `tensor.quantization`. The host data on `tensor.host` is left as the
/// original f32 reference so correctness checks can still compare against it.
pub(crate) fn apply_quantization(
    client: &ComputeClient<TestRuntime>,
    tensor: &mut TestTensor,
    scheme: QuantScheme,
) {
    let original_shape = tensor.handle.shape().clone();

    // Derive the scale tensor from the quant value's range. For
    // `QuantLevel::Tensor` a single scale is used; for `QuantLevel::Block` we
    // compute a per-block scale from the host data so each block fully uses
    // the quant range without clipping.
    let (scales_shape, scales_data) = compute_input_scales(&tensor.host, &scheme);
    let scale_handle = TestInput::builder(client.clone(), scales_shape.clone())
        .custom(scales_data)
        .generate();

    // Determine the correct storage type for the quantized output buffer.
    let output_storage_type = match &scheme.store {
        QuantStore::PackedU32(_) => StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
        QuantStore::PackedNative(_) | QuantStore::Native => {
            StorageType::Scalar(ElemType::from_quant_value(scheme.value))
        }
    };

    let mut quant_shape = original_shape.clone();
    let num_quants = scheme.num_quants();
    // Only divide last dim for PackedU32/PackedNative; Native stores 1:1.
    match &scheme.store {
        QuantStore::PackedU32(_) | QuantStore::PackedNative(_) => {
            if num_quants > 1 {
                let last_dim = quant_shape.len() - 1;
                quant_shape[last_dim] /= num_quants;
            }
        }
        QuantStore::Native => {}
    }

    let output_handle = TestInput::builder(client.clone(), quant_shape)
        .dtype(output_storage_type)
        .zeros()
        .generate();

    let out_scale_handle = TestInput::builder(client.clone(), scales_shape)
        .zeros()
        .generate();

    let input_elem = match tensor.handle.dtype {
        StorageType::Scalar(elem) => elem,
        _ => panic!("Unsupported storage type {:?}", tensor.handle.dtype),
    };

    cubek_quant::quantize::launch_ref(
        client,
        tensor.handle.clone().binding(),
        output_handle.clone().binding(),
        scale_handle.binding(),
        out_scale_handle.clone().binding(),
        &scheme,
        input_elem,
    )
    .expect("Quantization failed");

    // Keep the packed shape on the handle.
    tensor.handle = output_handle;
    tensor.quantization = Some(QuantizationInfo {
        scheme,
        scale: out_scale_handle,
        shape: original_shape,
    });
}

/// Compute the scale tensor shape and values for a quantized input.
///
/// For `QuantLevel::Tensor` this returns a single-element scale based on the
/// quant value's range (assumes input in [-1, 1]). For `QuantLevel::Block`
/// each block gets its own scale derived from `max(|value|)` in that block,
/// matching the reference pattern used by the cubek-quant symmetric tests.
fn compute_input_scales(host: &HostData, scheme: &QuantScheme) -> (Shape, Vec<f32>) {
    let (q_min, q_max) = scheme.value.range();
    let max_abs_q = q_max.abs().max(q_min.abs());

    match &scheme.level {
        QuantLevel::Tensor => {
            let shape: Shape = core::iter::repeat_n(1, host.shape.len()).collect();
            (shape, vec![1.0 / max_abs_q])
        }
        QuantLevel::Block(block_size) => {
            let rank = host.shape.len();
            let block_dims: Vec<usize> = block_size
                .to_dim_vec(rank)
                .into_iter()
                .map(|b| b as usize)
                .collect();

            let scales_shape: Shape = host
                .shape
                .iter()
                .zip(block_dims.iter())
                .map(|(d, b)| {
                    assert!(
                        d.is_multiple_of(*b),
                        "Block size {b} must divide dimension {d}",
                    );
                    d / b
                })
                .collect();

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

            (scales_shape, scales)
        }
    }
}
