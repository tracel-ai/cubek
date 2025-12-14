use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, prelude::*};

use crate::test_utils::test_tensor::test_input::base::{
    Distribution, RandomInputSpec, TestInputError,
};

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

pub(crate) fn build_random(
    spec: RandomInputSpec,
) -> Result<TensorHandle<TestRuntime>, TestInputError> {
    let shape = &spec.inner.shape;
    let strides = &spec.inner.stride_spec.compute_strides(shape);

    let handle = random_tensor_handle(
        &spec.inner.client,
        spec.inner.dtype,
        spec.seed,
        strides,
        shape,
        spec.distribution,
    );

    Ok(handle)
}
