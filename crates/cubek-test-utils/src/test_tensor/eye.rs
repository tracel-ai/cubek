use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{TensorHandle, ViewOperationsMut, ViewOperationsMutExpand},
};

use crate::BaseInputSpec;

#[cube(launch)]
fn eye_launch<T: Numeric>(tensor: &mut Tensor<Line<T>>, #[define(T)] _types: StorageType) {
    let batch = CUBE_POS_Z as usize;
    let i = ABSOLUTE_POS_X as usize;
    let j = ABSOLUTE_POS_Y as usize;

    let rank = tensor.rank();
    let rows = tensor.shape(rank - 2);
    let cols = tensor.shape(rank - 1);
    if i >= rows || j >= cols {
        terminate!();
    }

    let idx =
        batch * tensor.stride(rank - 3) + i * tensor.stride(rank - 2) + j * tensor.stride(rank - 1);

    tensor.write_checked(idx, Line::cast_from(i == j));
}

#[allow(unused)]
fn new_eyed(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    rows: usize,
    cols: usize,
    total_batches: usize,
    dtype: StorageType,
    strides: Vec<usize>,
) -> TensorHandle<TestRuntime> {
    // Performance is not important here and this simplifies greatly the problem
    let line_size = 1;

    let dim_x = 32;
    let dim_y = 32;
    let cube_dim = CubeDim::new_2d(dim_x, dim_y);
    let cube_count = CubeCount::new_3d(
        (rows as u32).div_ceil(dim_x),
        (cols as u32).div_ceil(dim_y),
        total_batches as u32,
    );

    let out = TensorHandle::new(
        client.empty(dtype.size() * shape.iter().product::<usize>()),
        shape.clone(),
        strides,
        dtype,
    );

    eye_launch::launch::<TestRuntime>(
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
        dtype,
    )
    .unwrap();

    out
}

pub(crate) fn build_eye(spec: BaseInputSpec) -> TensorHandle<TestRuntime> {
    let (batches, matrix) = spec.shape.split_at(spec.shape.len() - 2);
    let rows = matrix[0];
    let cols = matrix[1];
    let total_batches = batches.iter().product::<usize>();

    new_eyed(
        &spec.client,
        spec.shape.clone(),
        rows,
        cols,
        total_batches,
        spec.dtype,
        spec.strides(),
    )
}
