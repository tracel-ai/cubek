//! Kernel 1: prepass `D = rowsum(dO ⊙ O)`, shape `[B, H, N]`, fp32.
//!
//! One program instance per row. `O` and `dO` may be any float dtype; the
//! dot product accumulates in fp32 and `D` is always written as fp32: `D` is
//! subtracted from `dP` in the softmax Jacobian and that step is the most
//! numerically sensitive part of the backward.

use cubecl::{CubeDim, calculate_cube_count_elemwise, client::Client, prelude::*};

use super::io_dtype;
use crate::forward::definition::{AttentionGlobalTypes, AttentionSetupError};

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_prepass_kernel<E: Float>(
    o: &Tensor<E>,
    do_: &Tensor<E>,
    d: &mut Tensor<f32>,
    #[define(E)] _dtype: ElemType,
) {
    let row_idx = ABSOLUTE_POS;
    if row_idx >= d.len() {
        terminate!();
    }

    let head_dim = o.shape(o.rank() - 1);
    let base = row_idx * head_dim;

    let mut acc = 0.0_f32;
    for k in 0..head_dim {
        acc += f32::cast_from(o[base + k]) * f32::cast_from(do_[base + k]);
    }

    d[row_idx] = acc;
}

#[cube(launch, address_type = "dynamic")]
fn flash_attention_backward_prepass_plane_kernel<E: Float>(
    o: &Tensor<E>,
    do_: &Tensor<E>,
    d: &mut Tensor<f32>,
    #[define(E)] _dtype: ElemType,
) {
    let row_idx = CUBE_POS * (CUBE_DIM_Y as usize) + (UNIT_POS_Y as usize);
    if row_idx >= d.len() {
        terminate!();
    }

    let lane = UNIT_POS_X as usize;
    let head_dim = o.shape(o.rank() - 1);
    let base = row_idx * head_dim;

    let mut acc = 0.0_f32;
    let mut k = lane;
    while k < head_dim {
        acc += f32::cast_from(o[base + k]) * f32::cast_from(do_[base + k]);
        k += 32;
    }

    let sum = plane_sum(acc);
    if lane == 0 {
        d[row_idx] = sum;
    }
}

/// Compute `D = rowsum(dO ⊙ O)` into a pre-allocated fp32 tensor.
///
/// Inputs (dtype taken from `global_dtypes.out`):
/// - `o`:   `[B, H, N, d]`: output of the forward pass.
/// - `do_`: `[B, H, N, d]`: upstream gradient.
///
/// Output:
/// - `d`:   `[B, H, N]` fp32: written cleanly.
pub fn flash_attention_backward_prepass(
    client: &Client,
    o: TensorBinding,
    do_: TensorBinding,
    d: TensorBinding,
    global_dtypes: &AttentionGlobalTypes,
) -> Result<(), AttentionSetupError> {
    let dtype = io_dtype(global_dtypes)?;
    let f32_size = f32::elem_type_native().size();

    let working_units = d.shape.iter().product::<usize>();

    let address_type = o
        .required_address_type(dtype.size())
        .max(do_.required_address_type(dtype.size()))
        .max(d.required_address_type(f32_size));

    if client.properties().hardware.plane_size_max == 32 {
        let warps_per_cube = 4u32;
        let cube_dim = CubeDim::new_2d(32, warps_per_cube);
        let num_cubes = working_units.div_ceil(warps_per_cube as usize);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        flash_attention_backward_prepass_plane_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            o.into_tensor_arg(),
            do_.into_tensor_arg(),
            d.into_tensor_arg(),
            dtype,
        );
    } else {
        let cube_dim = CubeDim::new(client, working_units);
        let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
        flash_attention_backward_prepass_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            o.into_tensor_arg(),
            do_.into_tensor_arg(),
            d.into_tensor_arg(),
            dtype,
        );
    }

    Ok(())
}
