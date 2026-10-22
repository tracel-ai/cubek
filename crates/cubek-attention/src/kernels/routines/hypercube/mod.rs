//! The hypercube scope: several cubes own each box of queries, every cube walking its own run of
//! the keys with its planes, each plane its part of the run, and their partials merged by the last
//! cube to arrive or by a second launch ([`HypercubeSecondLaunch`]).

mod base;
mod fragment;
mod merge;
mod shuffle;

pub use base::*;
pub use fragment::*;
pub use merge::*;
pub use shuffle::*;
