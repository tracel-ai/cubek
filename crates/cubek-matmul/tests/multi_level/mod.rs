//! Tests for the routines written on the batch/global/stage/tile levels: the legacy kernels
//! cubek-convolution still builds on, held to their basic cases and their bias.

mod basic;
mod bias;

#[cfg(feature = "benchmarks")]
mod bench_catalog;
