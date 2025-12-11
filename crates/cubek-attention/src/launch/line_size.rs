use std::fmt::Debug;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
/// Line size used for each tensor in global memory accesses.
/// Represents the number of elements processed per SIMD load/store.
pub struct AttentionLineSizes {
    pub query: u8,
    pub key: u8,
    pub value: u8,
    pub mask: u8,
    pub out: u8,
}
