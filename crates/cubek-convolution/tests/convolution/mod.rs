//! A file per routine, each against the host's passes in [`base`].

mod base;
#[cfg(feature = "benchmarks")]
mod bench_catalog;
mod depthwise;
mod direct;
mod im2col;
mod multi_level;
mod split;
mod unit;

pub(crate) use base::*;
