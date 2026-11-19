use cubek_test_utils::{HostData, Progress};

use super::reference_ranked;

/// TopK returns the `k` largest values per output slice.
/// The output shape has `axis` set to `k` (rather than `1` as for scalar reductions).
pub fn reference_topk(
    input: &HostData,
    axis: usize,
    k: usize,
    progress: Option<&Progress>,
) -> HostData {
    reference_ranked(input, axis, k, progress, |pair| {
        pair.map_or(f32::MIN, |&(value, _)| value)
    })
}
