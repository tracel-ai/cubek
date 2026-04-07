use cubecl::{
    client::ComputeClient,
    std::tensor::TensorHandle,
    {TestRuntime, prelude::*},
};

use crate::{BaseInputSpec, Distribution};

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
        Distribution::Uniform(lower, upper) => cubek_random::random_uniform(
            client,
            lower,
            upper,
            tensor_handle.clone().binding(),
            dtype,
        )
        .unwrap(),
        Distribution::Bernoulli(prob) => {
            cubek_random::random_bernoulli(client, prob, tensor_handle.clone().binding(), dtype)
                .unwrap()
        }
    }

    tensor_handle
}

pub(crate) fn build_random(
    base_spec: BaseInputSpec,
    seed: u64,
    distribution: Distribution,
) -> TensorHandle<TestRuntime> {
    let shape = &base_spec.shape;
    let strides = &base_spec.strides();

    random_tensor_handle(
        &base_spec.client,
        base_spec.dtype,
        seed,
        strides,
        shape,
        distribution,
    )
}
