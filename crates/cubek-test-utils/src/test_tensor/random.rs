use cubecl::{
    client::ComputeClient,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
    {TestRuntime, prelude::*},
};

use crate::{BaseInputSpec, Distribution, test_tensor::strides::physical_extent};

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
    // Size the physical buffer to cover every logical index under these
    // strides — not just `shape.product()`. Jumpy strides (e.g. a slice that
    // steps over padding) need more room; broadcast strides (0) need less.
    let physical_len = physical_extent(&Shape::from(tensor_shape.to_vec()), &Strides::new(strides));
    let tensor_handle = TensorHandle::empty(client, vec![physical_len], dtype);

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
        Distribution::Normal { mean, std } => {
            cubek_random::random_normal(client, mean, std, tensor_handle.clone().binding(), dtype)
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
