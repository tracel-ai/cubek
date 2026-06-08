use crate::components::{NdLayoutLaunch, TestOp, resample_kernel};
use cubecl::prelude::*;

pub fn resample_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    out_layout: NdLayoutLaunch<R>,
    in_layout: NdLayoutLaunch<R>,
    scales: SequenceArg<R, f32>,
    dtype: StorageType,
) {
    let op = TestOp;

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_2d(32, 8),
            input.into_tensor_arg(),
            output.into_tensor_arg(),
            out_layout,
            in_layout,
            scales,
            op,
            dtype,
        )
    };
}
