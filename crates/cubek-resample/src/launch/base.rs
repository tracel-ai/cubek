use crate::{
    components::{NdLayoutLaunch, resample_kernel},
    definition::Resample,
};
use cubecl::prelude::*;

/// Launch the resample kernel for a single spatial axis.
pub fn resample_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    config: Resample,
    spatial_axis: usize,
    dtype: StorageType,
) {
    let out_layout = NdLayoutLaunch::from_tensor(&output);
    let in_layout = NdLayoutLaunch::from_tensor(&input);

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_2d(32, 8),
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            out_layout,
            in_layout,
            config,
            spatial_axis,
            dtype,
        );
    }
}
