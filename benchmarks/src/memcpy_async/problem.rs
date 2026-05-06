use crate::registry::CatalogEntry;

pub struct MemcpyAsyncProblem {
    pub data_count: usize,
    pub window_size: usize,
    pub double_buffering: bool,
}

pub fn problems() -> Vec<CatalogEntry<MemcpyAsyncProblem>> {
    vec![CatalogEntry::new(
        "data10m_window2k_double",
        "data=10M window=2048 double_buffering",
        MemcpyAsyncProblem {
            data_count: 10_000_000,
            window_size: 2048,
            double_buffering: true,
        },
    )]
}
