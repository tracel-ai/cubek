//! A batch of rows a row microkernel runs over.

/// `rows` rows, each `dim` wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowBatch {
    pub rows: usize,
    pub dim: usize,
}
