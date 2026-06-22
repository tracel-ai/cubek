pub mod components;
pub mod definition;
pub mod launch;

use crate::definition::{Resample, ResampleArgsLaunch};
use cubecl::prelude::*;

/// Resample an input tensor to produce an output tensor.
pub fn resample<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    args: ResampleArgsLaunch<R>,
    config: Resample,
    dtype: StorageType,
) {
    launch::resample_launch(client, input, output, args, config, dtype);
}
