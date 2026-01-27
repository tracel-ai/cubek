use cubecl::{
    TestRuntime, calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::{TensorHandle, ViewOperationsMut, ViewOperationsMutExpand},
};

use crate::BaseInputSpec;

#[cube(launch)]
fn custom_data_launch<T: Numeric>(
    tensor: &mut Tensor<T>,
    contiguous_data: Array<f32>,
    #[define(T)] _types: StorageType,
) {
    let linear = ABSOLUTE_POS;

    if linear >= tensor.len() {
        terminate!();
    }

    let mut remaining = linear;
    let mut offset = 0;

    for d in 0..tensor.rank() {
        let dim = tensor.shape(tensor.rank() - 1 - d);
        let idx = remaining % dim;
        remaining /= dim;
        offset += idx * tensor.stride(tensor.rank() - 1 - d);
    }

    tensor.write_checked(offset, T::cast_from(contiguous_data[linear]));
}

fn new_custom_data(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: StorageType,
    contiguous_data: Vec<f32>,
) -> TensorHandle<TestRuntime> {
    let num_elems = shape.iter().product::<usize>();
    assert!(contiguous_data.len() == num_elems);

    // Performance is not important here and this simplifies greatly the problem
    let line_size = 1;

    let working_units: u32 = num_elems as u32 / line_size as u32;
    let cube_dim = CubeDim::new(client, working_units as usize);
    let cube_count = calculate_cube_count_elemwise(client, working_units as usize, cube_dim);

    let out = TensorHandle::new(
        client.empty(dtype.size() * num_elems),
        shape,
        strides,
        dtype,
    );

    let contiguous_handle = client.create_from_slice(f32::as_bytes(&contiguous_data));

    custom_data_launch::launch::<TestRuntime>(
        client,
        cube_count,
        cube_dim,
        unsafe {
            TensorArg::from_raw_parts_and_size(
                &out.handle,
                &out.strides,
                &out.shape,
                line_size,
                dtype.size(),
            )
        },
        unsafe {
            ArrayArg::from_raw_parts_and_size(
                &contiguous_handle,
                num_elems,
                line_size,
                dtype.size(),
            )
        },
        dtype,
    )
    .unwrap();

    out
}

pub(crate) fn build_custom(
    base_spec: BaseInputSpec,
    contiguous_data: Vec<f32>,
) -> TensorHandle<TestRuntime> {
    new_custom_data(
        &base_spec.client,
        base_spec.shape.clone(),
        base_spec.strides(),
        base_spec.dtype,
        contiguous_data,
    )
}
