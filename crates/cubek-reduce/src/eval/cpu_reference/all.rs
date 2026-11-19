use cubek_test_utils::{HostData, Progress};

use super::reference_rows;

/// Logical-AND reference: 1.0 if every element along `axis` is non-zero, else 0.0.
pub fn reference_all(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    reference_rows(input, axis, progress, |row| {
        f32::from(u8::from(row.iter().all(|&value| value != 0.0)))
    })
}
