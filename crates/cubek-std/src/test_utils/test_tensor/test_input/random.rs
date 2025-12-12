use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, prelude::*};

use crate::test_utils::test_tensor::test_input::base::{
    Distribution, HostData, HostDataType, RandomInputSpec, TestInputError, TestInputResult,
};
use crate::test_utils::{batched_matrix_strides, copy_casted};

fn random_tensor_handle(
    client: &ComputeClient<TestRuntime>,
    dtype: StorageType,
    seed: u64,
    strides: &[usize],
    tensor_shape: &[usize],
    distribution: Distribution,
) -> TensorHandle<TestRuntime> {
    assert_eq!(tensor_shape.len(), strides.len());

    cubek_random::seed(seed);
    let flat_len: usize = tensor_shape.iter().product();
    let tensor_handle = TensorHandle::empty(client, vec![flat_len], dtype);

    match distribution {
        Distribution::Uniform(lower, upper) => {
            cubek_random::random_uniform(client, lower, upper, tensor_handle.as_ref(), dtype)
                .unwrap()
        }
        Distribution::Bernoulli(prob) => {
            cubek_random::random_bernoulli(client, prob, tensor_handle.as_ref(), dtype).unwrap()
        }
    }

    TensorHandle::new(
        tensor_handle.handle,
        tensor_shape.to_vec(),
        strides.to_vec(),
        tensor_handle.dtype,
    )
}

fn random_tensor_data<T: CubePrimitive + CubeElement + Default>(
    client: &ComputeClient<TestRuntime>,
    tensor_handle: &TensorHandle<TestRuntime>,
) -> Vec<T> {
    // Read data in row-major flat form
    let handle = copy_casted(client, tensor_handle, T::as_type_native_unchecked());
    T::from_bytes(&client.read_one_tensor(handle.as_copy_descriptor())).to_owned()
}

pub(crate) fn build_random(
    spec: RandomInputSpec,
    host_data_type: Option<HostDataType>,
) -> Result<TestInputResult, TestInputError> {
    let strides = &spec
        .inner
        .strides
        .unwrap_or(batched_matrix_strides(&spec.inner.shape, false));

    let handle = random_tensor_handle(
        &spec.inner.client,
        spec.inner.dtype,
        spec.seed,
        strides,
        &spec.inner.shape,
        spec.distribution,
    );

    let host_data = match host_data_type {
        Some(HostDataType::F32) => Some(HostData::F32(random_tensor_data(
            &spec.inner.client,
            &handle,
        ))),
        Some(HostDataType::Bool) => Some(HostData::Bool(
            random_tensor_data::<u8>(&spec.inner.client, &handle)
                .iter()
                .map(|&x| x > 0)
                .collect(),
        )),
        None => None,
    };

    Ok(TestInputResult { handle, host_data })
}
