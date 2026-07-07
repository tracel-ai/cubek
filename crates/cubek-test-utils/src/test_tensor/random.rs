use cubecl::{
    client::ComputeClient,
    std::tensor::TensorHandle,
    zspace::{Shape, Strides},
    {TestRuntime, prelude::*},
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{
    BaseInputSpec, Distribution,
    stubs::random::random_data,
    test_tensor::{custom::cast_f32_to_dtype, strides::physical_extent},
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

    // Size the physical buffer to cover every logical index under these
    // strides — not just `shape.product()`. Jumpy strides (e.g. a slice that
    // steps over padding) need more room; broadcast strides (0) need less.
    let physical_len = physical_extent(&Shape::from(tensor_shape.to_vec()), &Strides::new(strides));

    let mut rng = StdRng::seed_from_u64(seed);
    let data = random_data(&mut rng, distribution, physical_len);

    let bytes = cast_f32_to_dtype(&data, dtype);
    let handle = client.create_from_slice(&bytes);

    TensorHandle::new(
        handle,
        Shape::from(vec![physical_len]),
        Strides::new(&[1]),
        dtype,
    )
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
