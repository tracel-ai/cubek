pub mod basic;
#[cfg(feature = "benchmarks")]
pub mod bench_catalog;
#[cfg(feature = "extended")]
pub mod extended;
pub mod gqa;
pub mod lse;
pub mod masking;
pub mod sliding_window;
pub mod tail;

pub(crate) mod launcher;

pub(crate) use cubek_attention::eval::forward::cpu_reference::assert_result;
