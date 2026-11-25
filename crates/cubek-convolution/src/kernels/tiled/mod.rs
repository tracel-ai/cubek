//! A convolution over any number of spatial dims as one tile contraction over a gathered input: the
//! forward, the weight gradient and the data gradient share one space and one projection of the
//! input, with the roles of the dimensions moved ([`ConvolutionOperation`]). The data gradient is
//! at stride 1 only.
//!
//! Activations are channels last and the weight and its gradient `[co, taps.., ci]`. The routines
//! in `routines` are what runs, one folder per scope.

mod binding;
mod blueprint;
mod candidate;
mod decline;
mod key;
mod launch;
mod partitioning;
mod prepared;
mod problem;
mod roles;
mod routines;
mod staging;
mod tensors;

pub use candidate::CubeStagedConvCandidate;
pub use decline::ConvDecline;
pub use key::CubeStagedConvTuneKey;
pub use partitioning::ConvCoverage;
pub use prepared::CubeStagedConvPrepared;
pub use problem::SpatialDim;
pub use routines::{ConvRoutineKind, CubeStagedConvSettings};
pub use tensors::ConvTensors;

pub(crate) use binding::ConvBinding;
pub(crate) use blueprint::TileConvBlueprint;
pub(crate) use launch::ConvRoutine;
pub(crate) use partitioning::{ConvPartitioning, ConvTiles, InstructionShape};
pub(crate) use problem::{B, CI, CO, ConvDimension, ConvProblem, SpatialDimCount};
pub(crate) use roles::{ConvGather, ConvRoles};
pub(crate) use routines::CubeStagedConvRoutine;
pub(crate) use staging::ConvStaging;
