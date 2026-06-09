use crate::{components::resample_kernel, definition::Resample};
use cubecl::{
    prelude::*,
    std::{
        FastDivmod,
        tensor::{
            launch::ViewArg,
            layout::{
                CoordsDyn,
                fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
            },
        },
    },
};

/// Launch the resample kernel for a single spatial axis.
pub fn resample_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    config: Resample,
    dtype: StorageType,
) {
    let cube_count = CubeCount::Static(1, 1, 1);
    let cube_dim = CubeDim::new_2d(32, 8);

    let working_units = cube_dim.num_elems() as usize;

    let vector_size = 1;

    let out_shape = shape_divmod(&output);

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            view(input, vector_size),
            view(output, vector_size),
            out_shape,
            working_units,
            config,
            dtype,
        );
    }
}

fn shape_divmod<R: Runtime>(binding: &TensorBinding<R>) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

fn view<R: Runtime>(tensor: TensorBinding<R>, vector_size: VectorSize) -> ViewArg<CoordsDyn, R> {
    let shape = tensor
        .shape
        .iter()
        .map(|&s| s as u32)
        .collect::<SequenceArg<R, u32>>();

    let layout = FixedDimLayoutLaunch::<CoordsDyn, R>::from_shape_handle_unchecked(
        &tensor,
        shape,
        vector_size,
    );
    let buffer = tensor.into_tensor_arg();
    ViewArg::new_tensor::<FixedDimLayout<CoordsDyn>>(buffer, layout)
}
