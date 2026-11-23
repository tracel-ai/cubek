//! The workspace a test or a benchmark gives a reduction, as any caller of cubek does: cubek
//! never allocates one ([`ReduceWorkspace`]).

use crate::ReduceOperation;
use crate::kernels::{ReduceElements, ReduceWorkspace};
use cubecl::bytes::Bytes;
use cubecl::client::Client;

/// A workspace on `client` as large as [`ReduceWorkspace::bytes_for`] asks for reducing an
/// input of `shape` along `axis` by `operation` in `elements`, `with_indices` where the indices
/// are written beside the values, its counters zero.
///
/// # Panics
///
/// Where `axis` is not an axis of `shape`.
pub fn workspace_for(
    client: &Client,
    shape: &[usize],
    axis: usize,
    operation: ReduceOperation,
    elements: ReduceElements,
    with_indices: bool,
) -> ReduceWorkspace {
    let bytes = ReduceWorkspace::bytes_for(client, shape, axis, operation, elements, with_indices);
    // A buffer holds at least one counter's bytes: a runtime may refuse an empty allocation.
    let at_least_one = |bytes: u64| (bytes as usize).max(size_of::<u32>());
    let counters = at_least_one(bytes.arrivals) / size_of::<u32>();
    ReduceWorkspace {
        partials: client.empty(at_least_one(bytes.partials)),
        partial_indices: client.empty(at_least_one(bytes.partial_indices)),
        arrivals: client.create(Bytes::from_elems(vec![0u32; counters])),
    }
}
