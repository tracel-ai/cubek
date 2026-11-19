use cubek_test_utils::{HostData, Progress};

use super::{best_of, reference_rows, should_replace_max};

pub fn reference_argmax(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    reference_rows(input, axis, progress, |row| {
        best_of(row.iter().copied(), f32::NEG_INFINITY, should_replace_max).0 as f32
    })
}
