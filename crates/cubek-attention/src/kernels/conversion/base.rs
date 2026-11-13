//! An operand converted to another element ahead of a launch, into a buffer of its own: what a
//! routine reads at an element its stage cannot decode it into, where the operand is small
//! enough that one pass over it is nothing beside the attention. The query of a prompt chunk read
//! at its keys' `e4m3` is one: a few megabytes, against a cache of gigabytes.

use super::kernel::converted_kernel;
use cubecl::calculate_cube_count_elemwise;
use cubecl::client::Client;
use cubecl::ir::ElemType;
use cubecl::prelude::*;
use cubek_matmul::kernels::tensor::PlainTensor;

/// `tensor` converted to `element`, contiguous in a buffer of its own, every value cast.
pub(crate) fn converted(client: &Client, tensor: &PlainTensor, element: ElemType) -> PlainTensor {
    let shape = tensor.shape();
    let elements = shape.num_elements();
    let (_, converted) = PlainTensor::empty_contiguous(client, shape, element);
    if elements == 0 {
        return converted;
    }
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = calculate_cube_count_elemwise(client, elements, cube_dim);
    // SAFETY: every unit past `elements` returns before it reads; each reads `tensor` at its own
    // coordinate through the tensor's strides and writes its own element of `converted`.
    unsafe {
        converted_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            tensor.binding.clone().into_tensor_arg(),
            converted.binding.clone().into_tensor_arg(),
            u32::try_from(elements).expect("a converted operand counts its elements in a u32"),
            [tensor.elem, element],
        );
    }
    converted
}
