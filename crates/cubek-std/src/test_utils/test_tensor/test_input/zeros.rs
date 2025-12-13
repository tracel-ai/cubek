use cubecl::{TestRuntime, std::tensor::TensorHandle};

use crate::test_utils::test_tensor::test_input::base::{SimpleInputSpec, TestInputError};

pub(crate) fn build_zeros(
    spec: SimpleInputSpec,
) -> Result<TensorHandle<TestRuntime>, TestInputError> {
    if spec.strides.is_some() {
        return Err(TestInputError::UnsupportedStrides);
    }

    // let host_data = match host_data_type {
    //     Some(HostDataType::F32) => Some(HostData::F32(vec![0.0; num_elems])),
    //     Some(HostDataType::Bool) => Some(HostData::Bool(vec![false; num_elems])),
    //     None => None,
    // };

    Ok(TensorHandle::zeros(&spec.client, spec.shape, spec.dtype))
}
