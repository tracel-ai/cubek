use cubecl::{TestRuntime, std::tensor::TensorHandle};

use crate::BaseInputSpec;

pub(crate) fn build_zeros(spec: BaseInputSpec) -> TensorHandle<TestRuntime> {
    let mut tensor = TensorHandle::zeros(&spec.client, spec.shape.clone(), spec.dtype);

    // This manipulation is only valid since all the data is the same
    tensor.strides = spec.strides();

    tensor
}
