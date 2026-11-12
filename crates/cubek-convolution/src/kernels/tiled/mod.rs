//! A 2D convolution as one tile contraction over a gathered input: the forward, the weight
//! gradient and the data gradient share one space and one projection of the input, with the roles
//! of the axes moved ([`ConvolutionOperation`]). The data gradient is at stride 1 only.
//!
//! Activations are NHWC and the weight and its gradient `[co, kh, kw, ci]`. The routines in
//! `routines` are what runs, one folder per scope.

mod binding;
mod blueprint;
mod declined;
mod launch;
mod partitioning;
mod problem;
mod roles;
mod routines;
mod staging;
mod tensors;
mod tune;

pub use declined::ConvDeclined;
pub use partitioning::CMMA_STEP_ACROSS_TAPS;
pub use problem::ConvProblem;
pub use roles::{ConvContractedAxes, ConvExtents, ConvFreeAxes, ConvGather, ConvRoles, ViewDim};
pub use routines::{
    CONV_INSTRUCTION_UNOFFERED, CONV_WGMMA_UNSERVED, ConvRoutineKind, ConvSettings, ConvVectors,
    CubeStagedConvRoutine, UnitBlock, UnitRegistersConvRoutine, UnitRegistersConvSettings,
};
pub use tensors::ConvTensors;
pub use tune::ConvCandidate;

pub(crate) use binding::ConvBinding;
pub(crate) use blueprint::TileConvBlueprint;
pub(crate) use launch::ConvRoutine;
pub(crate) use partitioning::{ConvPartitioning, ConvTiles, InstructionShape};
pub(crate) use problem::{B, CI, CO, OH, OW, RH, RW};
pub(crate) use staging::ConvStaging;
pub(crate) use tune::launch_tuned;
