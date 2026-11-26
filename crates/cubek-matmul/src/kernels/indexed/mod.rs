//! The products whose operands read a table, one entry a batch
//! ([`MatmulIndex`](super::MatmulIndex)), state their inputs as the dense product's batches, an
//! index table and its stack read slab by slab ([`stack`]), and launch a dense routine on them.

pub mod stack;
