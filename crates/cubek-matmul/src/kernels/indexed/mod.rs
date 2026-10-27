//! The products whose operands read a table, one entry a batch ([`MatmulIndex`](super::MatmulIndex)):
//! none holds a routine or a kernel. Each states its inputs as the dense product's batches, an index
//! table and its stack read slab by slab ([`stack`]), and launches a dense routine on them.

pub mod k_grouped;
pub mod m_grouped;
pub mod routed;
pub mod stack;
