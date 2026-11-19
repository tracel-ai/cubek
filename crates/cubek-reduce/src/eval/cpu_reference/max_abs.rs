use cubek_test_utils::{HostData, Progress};

use super::{best_of, reference_rows, should_replace_max};

pub fn reference_max_abs(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    reference_rows(input, axis, progress, |row| {
        best_of(row.iter().map(|value| value.abs()), 0.0, should_replace_max).1
    })
}
