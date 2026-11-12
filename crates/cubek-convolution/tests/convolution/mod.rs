#![allow(missing_docs)]

#[allow(unused)]
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
mod split;
pub mod tile;
pub mod unit;

#[allow(unused)]
pub mod launcher_strategy;
