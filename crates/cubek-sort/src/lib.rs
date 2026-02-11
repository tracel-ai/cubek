//! CubeK Sort: Hardware-agnostic GPU radix sorting using CubeCL.

pub mod components;
pub mod launch;
pub mod routines;

mod error;

pub use components::config::SortStrategy;
pub use components::key::{Radix, SortKey};
pub use error::SortError;

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Sort order for radix sort operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortOrder {
    #[default]
    Ascending,
    Descending,
}

impl SortOrder {
    pub fn asc_or_desc(desc: bool) -> Self {
        if desc {
            SortOrder::Descending
        } else {
            SortOrder::Ascending
        }
    }

    pub fn is_descending(self) -> bool {
        matches!(self, SortOrder::Descending)
    }
}

/// Specifies how values should be handled during sorting.
pub enum SortValues<'a, R: Runtime> {
    /// No values - only sort keys.
    None,
    /// Sort key-value pairs together. Values are permuted alongside keys.
    Tensor(TensorHandleRef<'a, R>),
    /// Generate indices [0, 1, 2, ...] implicitly and sort them with keys.
    /// This is efficient for argsort operations - no input tensor allocation needed.
    Indices,
}

/// Output from a sort operation.
pub struct SortOutput {
    /// Sorted keys.
    pub keys: Handle,
    /// Sorted values. `None` if `SortValues::None` was used.
    pub values: Option<Handle>,
}

/// Sort keys with optional values.
///
/// Allocates output buffers internally and returns them in `SortOutput`.
///
/// # Arguments
/// * `keys_in` - Input keys to sort
/// * `values` - How to handle values: `None`, `Tensor`, or `Indices`
/// * `order` - Ascending or Descending
pub fn sort<'a, R: Runtime, K: SortKey>(
    client: &ComputeClient<R>,
    keys_in: TensorHandleRef<'a, R>,
    values: SortValues<'a, R>,
    order: SortOrder,
) -> Result<SortOutput, SortError>
where
    K::Radix: SortKey<Radix = K::Radix>,
{
    let has_values = !matches!(values, SortValues::None);
    let strategy = if has_values {
        SortStrategy::for_pairs(keys_in.shape[0])
    } else {
        SortStrategy::for_keys(keys_in.shape[0])
    };
    launch::sort::<R, K>(client, keys_in, values, strategy, order)
}
