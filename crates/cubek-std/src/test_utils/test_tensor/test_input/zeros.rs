use cubecl::std::tensor::TensorHandle;

use crate::test_utils::test_tensor::test_input::base::{
    HostData, HostDataType, SimpleInputSpec, TestInputError, TestInputResult,
};

pub(crate) fn build_zeros(
    spec: SimpleInputSpec,
    host_data_type: Option<HostDataType>,
) -> Result<TestInputResult, TestInputError> {
    let num_elems = spec.shape.iter().product();

    if spec.strides.is_some() {
        return Err(TestInputError::UnsupportedStrides);
    }

    let host_data = match host_data_type {
        Some(HostDataType::F32) => Some(HostData::F32(vec![0.0; num_elems])),
        Some(HostDataType::Bool) => Some(HostData::Bool(vec![false; num_elems])),
        None => None,
    };

    Ok(TestInputResult {
        handle: TensorHandle::zeros(&spec.client, spec.shape, spec.dtype),
        host_data,
    })
}
