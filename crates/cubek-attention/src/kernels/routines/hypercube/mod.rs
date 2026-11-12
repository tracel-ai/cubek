//! The hypercube scope: several cubes own each box of queries, every cube walking its own run of
//! the keys with its planes, each plane its part of the run. The last cube of a box to arrive
//! merges their partials into the output, or the routine leaves them in slots for the split-keys
//! merge launched after it ([`PlaneSplitKeysMergeRoutine`](crate::kernels::PlaneSplitKeysMergeRoutine)).
//! Or every box's keys lie on one line the cubes share in runs
//! ([`HypercubeStreamFragmentAttentionRoutine`]).

mod base;
mod fragment;
mod shuffle;
mod stream;

pub use base::*;
pub use fragment::*;
pub use shuffle::*;
pub use stream::*;
