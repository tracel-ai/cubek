use cubek_test_utils::{HostData, Progress};

use super::reference_ranked;

/// ArgTopK returns the `k` axis indices of the top values per output slice, laid out as
/// [`reference_topk`](super::reference_topk) lays out their values.
pub fn reference_argtopk(
    input: &HostData,
    axis: usize,
    k: usize,
    progress: Option<&Progress>,
) -> HostData {
    reference_ranked(input, axis, k, progress, |pair| {
        pair.map_or(u32::MAX, |&(_, index)| index) as f32
    })
}
