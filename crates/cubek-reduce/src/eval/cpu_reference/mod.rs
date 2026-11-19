//! CPU reference and seeded "produce a HostData" primitives for reduce.
//!
//! Both [`strategy_result`] and [`cpu_reference_result`] build the same input
//! bits from `(input, shape, axis, operation)` so the two `HostData`s they return
//! are directly comparable for the same `axis`/`operation`.

mod all;
mod any;
mod argmax;
mod argmin;
mod argtopk;
mod base;
mod max;
mod max_abs;
mod mean;
mod min;
mod prod;
mod sum;
mod topk;

pub use all::reference_all;
pub use any::reference_any;
pub use argmax::reference_argmax;
pub use argmin::reference_argmin;
pub use argtopk::reference_argtopk;
pub use base::*;
pub use max::reference_max;
pub use max_abs::reference_max_abs;
pub use mean::reference_mean;
pub use min::reference_min;
pub use prod::reference_prod;
pub use sum::reference_sum;
pub use topk::reference_topk;
