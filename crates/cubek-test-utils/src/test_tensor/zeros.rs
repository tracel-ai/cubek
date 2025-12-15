use cubecl::{TestRuntime, std::tensor::TensorHandle};

use crate::SimpleInputSpec;

pub(crate) fn build_zeros(spec: SimpleInputSpec) -> TensorHandle<TestRuntime> {
    let mut tensor = TensorHandle::zeros(&spec.client, spec.shape.clone(), spec.dtype);

    // This manipulation is only valid since all the data is the same
    tensor.strides = spec.strides();

    tensor
}
