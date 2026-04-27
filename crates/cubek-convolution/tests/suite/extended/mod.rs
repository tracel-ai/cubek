//! Extended tier: hand-written forced-blueprint tests covering harder or
//! niche cases — TilingScheme sweep on a representative algorithm, alt
//! convolution sizes, and advanced knobs (swizzle, partition buffering).
//!
//! Each axis is covered against one representative routine — the cartesian
//! lives in `full/`.

mod common;

mod advanced;
mod alt_shapes;
mod tiling_scheme;
