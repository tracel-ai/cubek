//! CubeK Sort: Hardware-agnostic GPU radix sorting using CubeCL.

pub mod components;
pub mod kernels;
pub mod launch;

mod error;

pub use components::config::SortStrategy;
pub use components::key::{Radix, SortKey};
pub use error::SortError;

use cubecl::prelude::*;

/// Sort order for radix sort operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    #[default]
    Ascending,
    Descending,
}

impl SortOrder {
    pub fn is_descending(self) -> bool {
        matches!(self, SortOrder::Descending)
    }
}

/// Sort keys in the specified order.
pub fn sort_keys<R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    num_items: usize,
    order: SortOrder,
) -> Result<(), SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    let strategy = SortStrategy::for_keys(num_items);
    launch::sort_keys::<R, K>(client, keys_in, keys_out, num_items, strategy, order)
}

/// Sort key-value pairs by key in the specified order (stable).
#[allow(clippy::too_many_arguments)]
pub fn sort_pairs<R: Runtime, K: SortKey, V: Numeric>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<R>,
    keys_out: TensorHandleRef<R>,
    values_in: TensorHandleRef<R>,
    values_out: TensorHandleRef<R>,
    num_items: usize,
    order: SortOrder,
) -> Result<(), SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    let strategy = SortStrategy::for_pairs(num_items);
    launch::sort_pairs::<R, K, V>(
        client, keys_in, keys_out, values_in, values_out, num_items, strategy, order,
    )
}
