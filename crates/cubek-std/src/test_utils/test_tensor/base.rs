use cubecl::client::ComputeClient;
use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, prelude::*};

use crate::test_utils::new_casted;

/// Returns random input tensor with arbitrary user-provided strides.
/// The returned Vec<f32> contains the same values but rearranged to match
/// the logical indexing implied by the strides.
pub fn random_tensor(
    client: &ComputeClient<TestRuntime>,
    dtype: StorageType,
    seed: u64,
    strides: &[usize],
    tensor_shape: &[usize],
) -> (TensorHandle<TestRuntime>, Vec<f32>) {
    assert_eq!(tensor_shape.len(), strides.len());

    // Create flattened random buffer
    cubek_random::seed(seed);
    let flat_len: usize = tensor_shape.iter().product();
    let tensor_handle = TensorHandle::empty(client, vec![flat_len], dtype);

    cubek_random::random_uniform(client, -1., 1., tensor_handle.as_ref(), dtype).unwrap();

    // Read data in row-major flat form
    let data_handle = new_casted(client, &tensor_handle, f32::as_type_native_unchecked());
    let flat_data =
        f32::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();

    // Now reorder to match the logical indexing implied by strides
    let logical_data = reorder_by_strides(&flat_data, tensor_shape, strides);

    (
        TensorHandle::new(
            tensor_handle.handle,
            tensor_shape.to_vec(),
            strides.to_vec(),
            tensor_handle.dtype,
        ),
        logical_data,
    )
}

pub fn random_bool_tensor(
    client: &ComputeClient<TestRuntime>,
    dtype: StorageType,
    seed: u64,
    strides: &[usize],
    tensor_shape: &[usize],
) -> (TensorHandle<TestRuntime>, Vec<bool>) {
    assert_eq!(tensor_shape.len(), strides.len());

    // Create flattened random buffer
    cubek_random::seed(seed);
    let flat_len: usize = tensor_shape.iter().product();
    let tensor_handle = TensorHandle::empty(client, vec![flat_len], dtype);

    cubek_random::random_bernoulli(client, 0.1, tensor_handle.as_ref(), dtype).unwrap();

    // Read data in row-major flat form
    let data_handle = new_casted(client, &tensor_handle, u8::as_type_native_unchecked());
    let flat_data =
        u8::from_bytes(&client.read_one_tensor(data_handle.as_copy_descriptor())).to_owned();

    // Now reorder to match the logical indexing implied by strides
    let logical_data = reorder_by_strides(&flat_data, tensor_shape, strides);

    (
        TensorHandle::new(
            tensor_handle.handle,
            tensor_shape.to_vec(),
            strides.to_vec(),
            tensor_handle.dtype,
        ),
        logical_data.iter().map(|&x| x > 0).collect(),
    )
}

fn reorder_by_strides<T: Copy + Default>(flat: &[T], shape: &[usize], strides: &[usize]) -> Vec<T> {
    let total = flat.len();
    let mut out = vec![T::default(); total];

    let rank = shape.len();
    let mut index = vec![0usize; rank];

    #[allow(clippy::needless_range_loop)]
    for logical_flat_idx in 0..total {
        // Compute multi-dim index in row-major order
        let mut remaining = logical_flat_idx;
        for d in (0..rank).rev() {
            let dim = shape[d];
            index[d] = remaining % dim;
            remaining /= dim;
        }

        // Compute physical offset using custom strides
        let mut physical = 0usize;
        for d in 0..rank {
            physical += index[d] * strides[d];
        }

        out[logical_flat_idx] = flat[physical];
    }

    out
}
