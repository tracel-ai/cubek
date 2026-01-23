use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{TensorHandle, ViewOperationsMut, ViewOperationsMutExpand},
};

use crate::test_tensor::base::BaseInputSpec;

#[cube(launch)]
fn arange_launch<T: Numeric>(tensor: &mut Tensor<T>, #[define(T)] _types: StorageType) {
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

    tensor.write_checked(offset, T::cast_from(linear));
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

    let working_units: u32 = num_elems as u32 / line_size as u32;
    let cube_dim = CubeDim::new(client, working_units as usize);
    let cube_count = working_units.div_ceil(cube_dim.num_elems());

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

pub(crate) fn build_arange(spec: BaseInputSpec) -> TensorHandle<TestRuntime> {
    new_arange(&spec.client, spec.shape.clone(), spec.strides(), spec.dtype)
}
