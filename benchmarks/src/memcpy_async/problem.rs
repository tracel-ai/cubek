use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_DEFAULT_DOUBLE: &str = "data10m_window2k_double";

pub struct MemcpyAsyncProblem {
    pub data_count: usize,
    pub window_size: usize,
    pub double_buffering: bool,
}

pub fn problems() -> Vec<ItemDescriptor> {
    vec![ItemDescriptor {
        id: PROBLEM_DEFAULT_DOUBLE.to_string(),
        label: "data=10M window=2048 double_buffering".to_string(),
    }]
}

pub(crate) fn problem_for(id: &str) -> Option<MemcpyAsyncProblem> {
    Some(match id {
        PROBLEM_DEFAULT_DOUBLE => MemcpyAsyncProblem {
            data_count: 10_000_000,
            window_size: 2048,
            double_buffering: true,
        },
        _ => return None,
    })
}
