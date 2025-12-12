use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, prelude::*};

use crate::test_utils::test_tensor::strides_utils::reorder_by_strides;
use crate::test_utils::test_tensor::test_input::base::{
    Distribution, HostData, HostDataType, RandomInputSpec, TestInputError, TestInputResult,
};
use crate::test_utils::{contiguous_strides, new_casted};

// fn random_f32_tensor(
//     client: &ComputeClient<TestRuntime>,
//     dtype: StorageType,
//     seed: u64,
//     strides: &[usize],
//     tensor_shape: &[usize],
// ) -> (TensorHandle<TestRuntime>, Vec<f32>) {
//     random_tensor(
//         client,
//         dtype,
//         seed,
//         strides,
//         tensor_shape,
//         |tensor_handle_ref| {
//             cubek_random::random_uniform(client, -1., 1., tensor_handle_ref, dtype).unwrap()
//         },
//     )
// }

// fn random_bool_tensor(
//     client: &ComputeClient<TestRuntime>,
//     dtype: StorageType,
//     seed: u64,
//     strides: &[usize],
//     tensor_shape: &[usize],
// ) -> (TensorHandle<TestRuntime>, Vec<bool>) {
//     let (tensor_handle, data) = random_tensor::<u8, _>(
//         client,
//         dtype,
//         seed,
//         strides,
//         tensor_shape,
//         |tensor_handle_ref| {
//             cubek_random::random_bernoulli(client, 0.1, tensor_handle_ref, dtype).unwrap()
//         },
//     );

//     (tensor_handle, data.iter().map(|&x| x > 0).collect())
// }

// fn random_tensor<T, F>(
//     client: &ComputeClient<TestRuntime>,
//     dtype: StorageType,
//     seed: u64,
//     strides: &[usize],
//     tensor_shape: &[usize],
//     random_function: F,
// ) -> (TensorHandle<TestRuntime>, Vec<T>)
// where
//     T: CubePrimitive + CubeElement + Default,
//     F: FnOnce(TensorHandleRef<TestRuntime>) -> (),
// {
//     let tensor_handle =
//         random_tensor_handle(client, dtype, seed, strides, tensor_shape, random_function);
//     let data = random_tensor_data::<T>(client, &tensor_handle, strides, tensor_shape);

//     (tensor_handle, data)
// }

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
    strides: &[usize],
    tensor_shape: &[usize],
) -> Vec<T> {
    // Read data in row-major flat form
    let data_handle = new_casted(client, tensor_handle, T::as_type_native_unchecked());
    let flat_data =
        T::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();

    // Now reorder to match the logical indexing implied by strides
    reorder_by_strides(&flat_data, tensor_shape, strides)
}

pub(crate) fn build_random(
    spec: RandomInputSpec,
    host_data_type: Option<HostDataType>,
) -> Result<TestInputResult, TestInputError> {
    let strides = &spec
        .inner
        .strides
        .unwrap_or(contiguous_strides(&spec.inner.shape, false));

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
            strides,
            &spec.inner.shape,
        ))),
        Some(HostDataType::Bool) => Some(HostData::Bool(
            random_tensor_data::<u8>(&spec.inner.client, &handle, strides, &spec.inner.shape)
                .iter()
                .map(|&x| x > 0)
                .collect(),
        )),
        None => None,
    };

    Ok(TestInputResult {
        handle,
        host_data,
    })
}
