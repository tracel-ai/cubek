use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{TensorHandle, ViewOperationsMut, ViewOperationsMutExpand},
};

#[cube(launch)]
fn eye_launch<T: Numeric>(tensor: &mut Tensor<Line<T>>, #[define(T)] _types: StorageType) {
    let batch = CUBE_POS_Z;
    let i = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let j = CUBE_POS_Y * CUBE_DIM_Y + UNIT_POS_Y;

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
pub fn new_eyed(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    let (batches, matrix) = shape.split_at(shape.len() - 2);
    let rows = matrix[0] as u32;
    let cols = matrix[1] as u32;
    let total_batches = batches.iter().product::<usize>() as u32;

    // Performance is not important here and this simplifies greatly the problem
    let line_size = 1;

    let dim_x = 32;
    let dim_y = 32;
    let cube_dim = CubeDim::new_2d(dim_x, dim_y);
    let cube_count = CubeCount::new_3d(rows.div_ceil(dim_x), cols.div_ceil(dim_y), total_batches);

    let out = TensorHandle::new_contiguous(
        shape.clone(),
        client.empty(dtype.size() * shape.iter().product::<usize>()),
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
