//! Host-side reimplementations of sibling crates' APIs.
//!
//! `cubek-test-utils` is used by the very crates whose functionality it needs
//! (e.g. `cubek-random`, `cubek-quant`). Depending on them directly would
//! create a dependency cycle and stop those crates from using the test utils
//! themselves. Instead, each stub here provides a small, self-contained,
//! device-free reimplementation of just the surface this crate relies on.
//!
//! Stubs are intentionally pure (no compute-kernel launches) so they stay fast
//! and unit-testable; the device-side glue that consumes them lives elsewhere
//! (e.g. [`crate::test_tensor`]).

pub(crate) mod quant;
pub(crate) mod random;
