//! Basic tier: smoke tests for each convolution algorithm family.
//!
//! Tests with no `cfg` attribute are the **light** subset — always compiled,
//! exercised on the CPU runtime in CI. Adding `#[cfg(feature = "basic")]`
//! gates the slower / more comprehensive smoke tests intended for GPU runs.
//!
//! The forced-blueprint sweeps along TilingScheme, swizzle, layouts, etc. live
//! in `extended/`. The full per-axis cartesian lives in `full/`.

mod common;

mod simple_async;
mod simple_cyclic;
mod simple_strided;
mod simple_tilewise;
mod specialized_tma;
mod tma;
