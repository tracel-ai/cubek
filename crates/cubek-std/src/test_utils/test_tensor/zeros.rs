use cubecl::{TestRuntime, std::tensor::TensorHandle};

use crate::test_utils::test_tensor::base::{SimpleInputSpec, TestInputError};

pub(crate) fn build_zeros(
    spec: SimpleInputSpec,
) -> Result<TensorHandle<TestRuntime>, TestInputError> {
    let mut tensor = TensorHandle::zeros(&spec.client, spec.shape.clone(), spec.dtype);

    // This manipulation is only valid since all the data is the same
    tensor.strides = spec.strides();

    Ok(tensor)
}
