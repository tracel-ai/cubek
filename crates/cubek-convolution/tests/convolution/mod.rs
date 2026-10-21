#![allow(missing_docs)]

pub mod basic;
#[cfg(feature = "benchmarks")]
pub mod bench_catalog;
pub mod depthwise;
pub mod depthwise_weight;
pub mod direct;
#[cfg(feature = "extended")]
pub mod extended;
#[cfg(feature = "full")]
pub mod full;
pub mod im2col;
pub mod unit;

pub mod launcher_strategy;
