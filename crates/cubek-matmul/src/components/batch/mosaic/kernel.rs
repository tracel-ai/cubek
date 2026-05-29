//! The Mosaic kernel.
//!
//! The launched kernel is a single call, `c.mma(a, b)` — the tile-DSL
//! [`mma`](cubek_tile::Tile::mma) primitive, driven by the accumulator. Mosaic
//! is just the launch wrapper; the lowering lives in the axis-agnostic
//! [`cubek_tile`] engine.
//!
//! There is no staging. Each leaf is read straight from global memory — which
//! is exactly a contiguous block when the operands are stored in a tiled
//! layout, so the CPU loads a whole tile at once. Batch is free: a tile's
//! leading axes are batch axes (pinned by `partition`), the trailing two are the
//! matrix, and the contraction always runs on the 2-D leaf.
#![allow(non_snake_case)]

use cubecl::prelude::*;

// Glob brings the tile-DSL items *and* the cube-macro-generated `*Expand`
// companions the kernel below needs.
use cubek_tile::*;

/// The Mosaic kernel: every operand is a [`Tile`] (a semantic view + its space +
/// partitioner), so the whole matmul is one line, driven by the accumulator.
#[cube(launch)]
pub fn mosaic_kernel<E: Numeric, S: Size>(
    a: &Tile<'_, E, S>,
    b: &Tile<'_, E, S>,
    c: &mut Tile<'_, E, S>,
    #[define(E)] _dtype: StorageType,
    #[define(S)] _vector_size: usize,
) {
    c.mma(a, b);
}
