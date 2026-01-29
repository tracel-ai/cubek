//! CubeK Sort: Hardware-agnostic GPU radix sorting using CubeCL.
//!
//! This crate provides a portable LSD (least-significant-digit) radix sort implementation
//! based on the device radix sort from [b0nes164](https://github.com/b0nes164/GPUSorting).
//!
//! # Features
//!
//! - **Stable sorting**: Values with the same key preserve their original order
//! - **Key-value pairs**: Sort pairs by key while maintaining value associations
//! - **Multiple types**: Supports u32, i32, and f32 keys
//! - **Hardware agnostic**: Works across CUDA, WebGPU, and other CubeCL backends
//!
//! # Example
//!
//! ```ignore
//! use cubek_sort::{sort_keys, SortStrategy};
//!
//! let client = /* ... */;
//! let keys_in = /* input tensor handle */;
//! let keys_out = /* output tensor handle */;
//!
//! sort_keys::<Runtime, u32>(&client, keys_in, keys_out, num_items, None)?;
//! ```

pub mod components;
pub mod kernels;
pub mod launch;

mod error;

pub use components::config::{KeyTransform, KeyValueMode, SortBlueprint, SortStrategy};
pub use components::key::SortKey;
pub use error::SortError;

use cubecl::prelude::*;

/// Sort keys in ascending order.
///
/// # Arguments
///
/// * `client` - The compute client for the target runtime
/// * `keys_in` - Input tensor containing keys to sort
/// * `keys_out` - Output tensor for sorted keys
/// * `num_items` - Number of items to sort
/// * `strategy` - Optional strategy configuration; uses defaults if None
///
/// # Returns
///
/// Returns `Ok(())` on success, or a `SortError` on failure.
pub fn sort_keys<R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: Option<SortStrategy>,
) -> Result<(), SortError> {
    let strategy = strategy.unwrap_or_default();
    launch::sort_keys::<R, K>(client, keys_in, keys_out, num_items, strategy)
}

/// Sort key-value pairs by key in ascending order (stable).
///
/// This is a stable sort, meaning that values with equal keys will maintain
/// their relative order from the input.
///
/// # Arguments
///
/// * `client` - The compute client for the target runtime
/// * `keys_in` - Input tensor containing keys
/// * `keys_out` - Output tensor for sorted keys
/// * `values_in` - Input tensor containing values
/// * `values_out` - Output tensor for sorted values
/// * `num_items` - Number of items to sort
/// * `strategy` - Optional strategy configuration; uses defaults if None
///
/// # Returns
///
/// Returns `Ok(())` on success, or a `SortError` on failure.
pub fn sort_pairs<R: Runtime, K: SortKey, V: Numeric>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    values_in: TensorHandleRef<R>,
    values_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: Option<SortStrategy>,
) -> Result<(), SortError> {
    let strategy = strategy.unwrap_or_default();
    launch::sort_pairs::<R, K, V>(
        client, keys_in, keys_out, values_in, values_out, num_items, strategy,
    )
}
