pub mod key;
pub mod launch;
pub mod routines;

mod error;

pub use error::SortError;

pub use launch::sort;

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
