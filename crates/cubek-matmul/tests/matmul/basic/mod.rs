//! Normal tier: Inferred-blueprint tests for each routine family.
//!
//! These tests route through `launch_ref` with `BlueprintStrategy::Inferred`,
//! exercising the selector heuristic for each Strategy variant. One test per
//! (routine, backend) is typically enough — the forced-blueprint TilingScheme
//! sweep lives in the `extended` tier.

mod common;

mod auto;
mod gemv;
mod naive;
mod plane_accelerated;
mod plane_vecmat;
mod tma;
mod unit;
