use cubecl::{TestRuntime, std::tensor::TensorHandle};

use crate::test_utils::test_tensor::test_input::{
    base::{SimpleInputSpec, TestInputError},
    strides::StrideSpec,
};

pub(crate) fn build_zeros(
    spec: SimpleInputSpec,
) -> Result<TensorHandle<TestRuntime>, TestInputError> {
    if spec.stride_spec != StrideSpec::RowMajor {
        return Err(TestInputError::UnsupportedStrides);
    }

    Ok(TensorHandle::zeros(&spec.client, spec.shape, spec.dtype))
}
