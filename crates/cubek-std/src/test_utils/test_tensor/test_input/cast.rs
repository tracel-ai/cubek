use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{
        TensorHandle, ViewOperations, ViewOperationsExpand, ViewOperationsMut,
        ViewOperationsMutExpand,
    },
    tensor_line_size_parallel,
};

#[cube(launch)]
fn cast_launch<From: Numeric, To: Numeric>(
    from: &Tensor<Line<From>>,
    to: &mut Tensor<Line<To>>,
    #[define(From, To)] _types: [StorageType; 2],
) {
    cast_inner::<From, To>(from, to);
}

#[cube]
fn cast_inner<From: Numeric, To: Numeric>(from: &Tensor<Line<From>>, to: &mut Tensor<Line<To>>) {
    to.write_checked(
        ABSOLUTE_POS,
        Line::cast_from(from.read_checked(ABSOLUTE_POS)),
    )
}

pub fn copy_casted(
    client: &ComputeClient<TestRuntime>,
    original: &TensorHandle<TestRuntime>,
    target_type: StorageType,
) -> TensorHandle<TestRuntime> {
    if target_type == original.dtype {
        return TensorHandle::new_contiguous(
            original.shape.clone(),
            original.handle.clone(),
            target_type,
        );
    }

    let num_elems: usize = original.shape.iter().product();

    let line_size = tensor_line_size_parallel(
        TestRuntime::supported_line_sizes().iter().copied(),
        &[num_elems],
        &[1],
        0,
    );

    let num_units_needed: u32 = num_elems as u32 / line_size as u32;
    let cube_dim = CubeDim::default();
    let cube_count = num_units_needed.div_ceil(cube_dim.num_elems());

    let out = TensorHandle::new_contiguous(
        original.shape.clone(),
        client.empty(target_type.size() * num_elems),
        target_type,
    );

    cast_launch::launch::<TestRuntime>(
        client,
        CubeCount::Static(cube_count, 1, 1),
        cube_dim,
        original.as_arg(line_size),
        out.as_arg(line_size),
        [original.dtype, target_type],
    )
    .unwrap();

    out
}
