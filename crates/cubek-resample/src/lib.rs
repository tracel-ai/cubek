pub mod components;
pub mod definition;
pub mod launch;

use crate::components::NdLayoutLaunch;
use crate::launch::resample_launch;
use cubecl::prelude::*;

pub fn resample<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    scales: SequenceArg<R, f32>,
    dtype: StorageType,
) {
    let mut in_divmods = SequenceArg::new();
    let mut in_strides = SequenceArg::new();
    for i in 0..input.shape.len() {
        in_divmods.push(input.shape[i]);
        in_strides.push(input.strides[i]);
    }
    let in_layout = NdLayoutLaunch::new(in_divmods, in_strides);

    let mut out_divmods = SequenceArg::new();
    let mut out_strides = SequenceArg::new();
    for i in 0..output.shape.len() {
        out_divmods.push(output.shape[i]);
        out_strides.push(output.strides[i]);
    }
    let out_layout = NdLayoutLaunch::new(out_divmods, out_strides);

    resample_launch(
        client,
        input,
        output.clone(),
        out_layout,
        in_layout,
        scales,
        dtype,
    );
}
