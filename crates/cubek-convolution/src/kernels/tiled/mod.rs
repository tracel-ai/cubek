//! A convolution over any number of spatial dims as one tile contraction over a gathered input: the
//! forward, the weight gradient and the data gradient share one space and one projection of the
//! input, with the roles of the dimensions moved ([`ConvolutionOperation`]). The data gradient is
//! at stride 1 only.
//!
//! Activations are channels last and the weight and its gradient `[co, taps.., ci]`. The routines
//! in `routines` are what runs, one folder per scope.

mod bias;
mod binding;
mod blueprint;
mod decline;
mod launch;
mod partitioning;
mod problem;
mod roles;
mod routines;
mod staging;
mod tensors;
mod tune;

pub use decline::ConvDecline;
pub use problem::ConvBias;
pub use routines::{ConvRoutineKind, CubeStagedConvSettings};
pub use tensors::ConvTensors;
pub use tune::CubeStagedConvCandidate;

pub(crate) use bias::BiasTile;
pub(crate) use binding::ConvBinding;
pub(crate) use blueprint::TileConvBlueprint;
pub(crate) use launch::ConvRoutine;
pub(crate) use partitioning::{ConvPartitioning, ConvTiles, InstructionShape};
pub(crate) use problem::{B, CI, CO, ConvDimension, ConvProblem, G, SpatialDim, SpatialDimCount};
pub(crate) use roles::{ChannelAxis, ConvChannelOperands, ConvGather, ConvGrouping, ConvRoles};
pub(crate) use routines::{CubeStagedConvRoutine, PlaneWindowConvRoutine};
pub(crate) use staging::ConvStaging;
pub(crate) use tune::launch_tuned;
