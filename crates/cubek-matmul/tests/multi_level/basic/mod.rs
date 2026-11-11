//! Normal tier: Inferred-blueprint tests for each routine family.
//!
//! These tests route through `launch_ref` with `BlueprintStrategy::Inferred`,
//! exercising the selector heuristic for each Strategy variant: one test per
//! (routine, backend).

mod gemv;
mod naive;
mod plane_accelerated;
mod plane_vecmat;
mod tma;
mod unit;
