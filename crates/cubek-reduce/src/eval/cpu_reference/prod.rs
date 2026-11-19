use cubek_test_utils::{HostData, Progress};

use super::reference_rows;

pub fn reference_prod(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    reference_rows(input, axis, progress, |row| {
        row.iter().fold(1.0, |acc, value| acc * value)
    })
}
