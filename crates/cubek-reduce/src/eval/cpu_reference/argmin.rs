use cubek_test_utils::{HostData, Progress};

use super::{best_of, reference_rows, should_replace_min};

pub fn reference_argmin(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    reference_rows(input, axis, progress, |row| {
        best_of(row.iter().copied(), f32::INFINITY, should_replace_min).0 as f32
    })
}
