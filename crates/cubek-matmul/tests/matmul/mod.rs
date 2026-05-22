#![allow(missing_docs)]

pub mod basic;
#[cfg(feature = "benchmarks")]
pub mod bench_catalog;
#[cfg(feature = "extended")]
pub mod extended;
#[cfg(feature = "full")]
pub mod full;

mod bias;
pub(crate) mod launcher_strategy;

pub(crate) use launcher_strategy::test_matmul_strategy;

#[cfg(feature = "extended")]
pub(crate) use extended::test_matmul_test_strategy;

pub(crate) use cubek_matmul::eval::cpu_reference::assert_result;
