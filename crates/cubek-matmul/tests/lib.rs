#![allow(unused, clippy::upper_case_acronyms)]

mod harness;

mod auto;
#[cfg(feature = "multi-level")]
mod multi_level;
#[cfg(feature = "tiled")]
mod tiled;
