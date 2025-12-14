use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{TensorHandle, ViewOperationsMut, ViewOperationsMutExpand},
};

use crate::test_utils::{
    batched_matrix_strides,
    test_tensor::test_input::base::{SimpleInputSpec, TestInputError},
};

#[cube(launch)]
fn arange_launch<T: Numeric>(tensor: &mut Tensor<T>, #[define(T)] _types: StorageType) {
    tensor.write_checked(ABSOLUTE_POS, T::cast_from(ABSOLUTE_POS));
}

fn new_arange(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    let num_elems = shape.iter().product::<usize>();

    // Performance is not important here and this simplifies greatly the problem
    let line_size = 1;

    let num_units_needed: u32 = num_elems as u32 / line_size as u32;
    let cube_dim = CubeDim::default();
    let cube_count = num_units_needed.div_ceil(cube_dim.num_elems());

    let out = TensorHandle::new(
        client.empty(dtype.size() * num_elems),
        shape,
        strides,
        dtype,
    );

    arange_launch::launch::<TestRuntime>(
        client,
        CubeCount::new_1d(cube_count),
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
        dtype,
    )
    .unwrap();

    out
}

pub(crate) fn build_arange(
    spec: SimpleInputSpec,
) -> Result<TensorHandle<TestRuntime>, TestInputError> {
    let strides = spec
        .strides
        .unwrap_or(batched_matrix_strides(&spec.shape, false));

    Ok(new_arange(&spec.client, spec.shape, strides, spec.dtype))
}
