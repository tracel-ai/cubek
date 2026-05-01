//! The CPU reference now lives in `cubek_matmul::cpu_reference` (gated by the
//! `cpu-reference` feature, enabled in dev-dependencies). This module just
//! re-exports the bits the test suite uses so call sites stay unchanged.

pub use cubek_matmul::cpu_reference::{assert_result, matmul_cpu_reference};
