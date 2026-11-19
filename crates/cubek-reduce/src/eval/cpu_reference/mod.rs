//! CPU reference and seeded "produce a HostData" primitives for reduce.
//!
//! Both [`strategy_result`] and [`cpu_reference_result`] build the same input
//! bits from `(input, shape, axis, operation)` so the two `HostData`s they return
//! are directly comparable for the same `axis`/`operation`.

mod base;

pub use base::*;
