use cubecl::{
    TestRuntime,
    prelude::*,
    std::tensor::{
        TensorHandle, ViewOperations, ViewOperationsExpand, ViewOperationsMut,
        ViewOperationsMutExpand,
    },
};

/// Copy the logical contents of a strided tensor into a contiguous buffer,
/// casting element-wise to `To`.
///
/// The kernel iterates logical linear positions and maps each one to the
/// source's physical offset via its strides, so non-contiguous layouts
/// (col-major, jumpy, etc.) are gathered correctly. Not vectorised: test
/// utilities don't need peak throughput, and vector reads aren't safe for
/// arbitrary strides.
#[cube(launch)]
fn cast_launch<From: Numeric, To: Numeric>(
    from: &Tensor<From>,
    to: &mut Tensor<To>,
    #[define(From, To)] _types: [StorageType; 2],
) {
    let linear = ABSOLUTE_POS;
    if linear >= to.len() {
        terminate!();
    }

    // Decompose `linear` into a logical coordinate (last dim first), then
    // accumulate the source offset using the input strides.
    let mut remaining = linear;
    let mut from_offset = 0usize;
    for d in 0..from.rank() {
        let axis = from.rank() - 1 - d;
        let dim = from.shape(axis);
        let idx = remaining % dim;
        remaining /= dim;
        from_offset += idx * from.stride(axis);
    }

    to.write_checked(linear, To::cast_from(from.read_checked(from_offset)));
}

/// Copy `original` into a contiguous buffer, casting to `target_type` if needed.
///
/// The returned handle always has contiguous strides, regardless of the input
/// layout. Callers (notably [`HostData::from_tensor_handle`]) rely on this: the
/// previous fast path reinterpreted the source handle as contiguous without
/// copying, which silently dropped values for inputs with non-contiguous
/// strides (e.g. jumpy strides where the physical buffer is larger than the
/// logical element count).
pub fn copy_casted(
    client: &ComputeClient<TestRuntime>,
    original: TensorHandle<TestRuntime>,
    target_type: StorageType,
) -> TensorHandle<TestRuntime> {
    let input_contiguous = is_contiguous(original.shape(), original.strides());

    // Fast path: already contiguous and same dtype. Reinterpreting the handle
    // is safe here because the physical buffer contains the logical values in
    // the expected order.
    if target_type == original.dtype && input_contiguous {
        return TensorHandle::new_contiguous(
            original.shape().clone(),
            original.handle.clone(),
            target_type,
        );
    }

    let num_elems: usize = original.shape().num_elements();

    let working_units = num_elems as u32;
    let cube_dim = CubeDim::new(client, working_units as usize);
    let cube_count = working_units.div_ceil(cube_dim.num_elems());

    let out = TensorHandle::new_contiguous(
        original.shape().clone(),
        client.empty(target_type.size() * num_elems),
        target_type,
    );

    let dtype = original.dtype;

    cast_launch::launch::<TestRuntime>(
        client,
        CubeCount::Static(cube_count, 1, 1),
        cube_dim,
        original.into_arg(),
        out.clone().into_arg(),
        [dtype, target_type],
    );

    out
}

fn is_contiguous(shape: &cubecl::zspace::Shape, strides: &cubecl::zspace::Strides) -> bool {
    let n = shape.len();
    let mut expected: usize = 1;
    for i in (0..n).rev() {
        if shape[i] == 1 {
            continue;
        }
        if strides[i] != expected {
            return false;
        }
        expected *= shape[i];
    }
    true
}
