//! CubeK Sort: Hardware-agnostic GPU radix sorting using CubeCL.

pub mod components;
pub mod kernels;
pub mod launch;

mod error;

pub use components::config::SortStrategy;
pub use components::key::SortKey;
pub use error::SortError;

use cubecl::prelude::*;

/// Sort keys in ascending order.
pub fn sort_keys<R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: Option<SortStrategy>,
) -> Result<(), SortError> {
    launch::sort_keys::<R, K>(
        client,
        keys_in,
        keys_out,
        num_items,
        strategy.unwrap_or_default(),
    )
}

/// Sort key-value pairs by key in ascending order (stable).
pub fn sort_pairs<R: Runtime, K: SortKey, V: Numeric>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    values_in: TensorHandleRef<R>,
    values_out: TensorHandleRef<R>,
    num_items: usize,
    strategy: Option<SortStrategy>,
) -> Result<(), SortError> {
    launch::sort_pairs::<R, K, V>(
        client,
        keys_in,
        keys_out,
        values_in,
        values_out,
        num_items,
        strategy.unwrap_or_default(),
    )
}
