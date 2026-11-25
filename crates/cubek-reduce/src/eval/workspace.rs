use crate::ReduceOperation;
use crate::kernels::{ReduceElements, ReduceWorkspace};
use cubecl::bytes::Bytes;
use cubecl::client::Client;

/// The workspace a test or a benchmark gives a reduction, as any caller of cubek does: cubek
/// allocates none.
#[derive(Debug, Clone)]
pub struct EvalWorkspace {
    pub workspace: ReduceWorkspace,
}

impl EvalWorkspace {
    /// A workspace on `client` as large as [`ReduceWorkspace::bytes_for`] asks, its counters zero.
    ///
    /// # Panics
    ///
    /// As [`ReduceWorkspace::bytes_for`].
    pub fn new(
        client: &Client,
        shape: &[usize],
        axis: usize,
        operation: ReduceOperation,
        elements: ReduceElements,
        with_indices: bool,
    ) -> Self {
        let bytes =
            ReduceWorkspace::bytes_for(client, shape, axis, operation, elements, with_indices);
        // A buffer holds at least one counter's bytes: a runtime may refuse an empty allocation.
        let at_least_one = |bytes: u64| (bytes as usize).max(size_of::<u32>());
        let counters = at_least_one(bytes.arrivals_bytes) / size_of::<u32>();
        Self {
            workspace: ReduceWorkspace {
                partials: client.empty(at_least_one(bytes.partials_bytes)),
                partial_indices: client.empty(at_least_one(bytes.partial_indices_bytes)),
                arrivals: client.create(Bytes::from_elems(vec![0u32; counters])),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::arrivals_of;
    use cubecl::ir::{ElemType, FloatKind};

    #[test]
    fn its_counters_hold_zero() {
        let client = cubecl::test_device().client();
        let operation = ReduceOperation::Max;
        let elements =
            ReduceElements::for_operation(operation, ElemType::Float(FloatKind::F32), None);
        let given = EvalWorkspace::new(&client, &[3, 70001], 1, operation, elements, false);
        let counts = arrivals_of(&client, given.workspace);
        assert!(counts.iter().all(|count| *count == 0));
    }
}
