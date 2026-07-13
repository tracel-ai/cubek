//! Kernel 1: prepass `D = rowsum(dO ⊙ O)`, shape `[B, H, N]`, fp32.
//!
//! One program instance per row (or row-block). Output dtype must be fp32 —
//! `D` is subtracted from `dP` in the softmax Jacobian and that step is the
//! most numerically sensitive part of the backward.

use cubecl::{CubeDim, Runtime, calculate_cube_count_elemwise, client::ComputeClient, prelude::*};

use crate::forward::definition::AttentionSetupError;

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_prepass_kernel<E: Float>(
    o: &Tensor<E>,
    do_: &Tensor<E>,
    d: &mut Tensor<E>,
    #[define(E)] _dtype: StorageType,
) {
    let row_idx = ABSOLUTE_POS;
    if row_idx >= d.len() {
        terminate!();
    }

    let head_dim = o.shape(o.rank() - 1);
    let base = row_idx * head_dim;

    let mut acc = E::new(0.0_f32);
    for k in 0..head_dim {
        acc += o[base + k] * do_[base + k];
    }

    d[row_idx] = acc;
}

/// Compute `D = rowsum(dO ⊙ O)` into a pre-allocated fp32 tensor.
///
/// Inputs:
/// - `o`:   `[B, H, N, d]` — output of the forward pass.
/// - `do_`: `[B, H, N, d]` — upstream gradient.
///
/// Output:
/// - `d`:   `[B, H, N]` fp32 — written cleanly.
pub fn flash_attention_backward_prepass<R: Runtime>(
    client: &ComputeClient<R>,
    o: TensorBinding<R>,
    do_: TensorBinding<R>,
    d: TensorBinding<R>,
) -> Result<(), AttentionSetupError> {
    let dtype = f32::as_type_native_unchecked().storage_type();

    let working_units = d.shape.iter().product::<usize>();
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = o
        .required_address_type(dtype.size())
        .max(do_.required_address_type(dtype.size()))
        .max(d.required_address_type(dtype.size()));

    flash_attention_backward_prepass_kernel::launch::<R>(
        client,
        cube_count,
        cube_dim,
        address_type,
        o.into_tensor_arg(),
        do_.into_tensor_arg(),
        d.into_tensor_arg(),
        dtype,
    );

    Ok(())
}
