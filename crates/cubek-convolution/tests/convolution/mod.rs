#![allow(missing_docs)]

pub mod basic;
#[cfg(feature = "benchmarks")]
pub mod bench_catalog;
#[cfg(feature = "extended")]
pub mod extended;
#[cfg(feature = "full")]
pub mod full;

pub mod launcher_strategy;
