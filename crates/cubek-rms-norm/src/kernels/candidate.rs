//! A routine of the RMS-norm family at one of its settings, whichever routine it is
//! ([`RmsNormCandidate`]), and that routine and setting as a value ([`RmsNormRoutineSettings`]).

use crate::kernels::{
    CubeGroupsRmsNormRoutine, CubeGroupsRmsNormRoutineSettings, CubeRmsNormRoutine,
    CubeRmsNormRoutineSettings, PlaneRmsNormRoutine, PlaneRmsNormRoutineSettings, RmsNormCall,
    RmsNormRoutineCandidate,
};
use cubecl::tile::launch::{Decline, SinkLaunchVisitor};
use cubek_matmul::kernels::tensor::LhsOutput;

/// A routine of this kind at one of its settings, as a value: what a fused tail's election hashes,
/// and what names a candidate ([`RmsNormCandidate::settings`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RmsNormRoutineSettings {
    /// [`CubeGroupsRmsNormRoutine`], which serves every float row on every device.
    CubeGroups(CubeGroupsRmsNormRoutineSettings),
    /// [`PlaneRmsNormRoutine`], a candidate.
    Plane(PlaneRmsNormRoutineSettings),
    /// [`CubeRmsNormRoutine`], a candidate.
    Cube(CubeRmsNormRoutineSettings),
}

/// A routine of this family at one of its settings, whichever routine it is: what a caller that
/// measures the routines identifies a candidate by, and hands back for the one it found fastest, so
/// a launch outside a round reads the launches the winner prepared in it.
#[derive(Debug, Clone)]
pub enum RmsNormCandidate {
    /// [`CubeGroupsRmsNormRoutine`], which serves every float row on every device.
    CubeGroups(RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine>),
    /// [`PlaneRmsNormRoutine`], a candidate.
    Plane(RmsNormRoutineCandidate<PlaneRmsNormRoutine>),
    /// [`CubeRmsNormRoutine`], a candidate.
    Cube(RmsNormRoutineCandidate<CubeRmsNormRoutine>),
}

impl RmsNormCandidate {
    /// The routine this candidate launches, at its setting.
    pub fn settings(&self) -> RmsNormRoutineSettings {
        match self {
            Self::CubeGroups(candidate) => (*candidate.settings()).into(),
            Self::Plane(candidate) => (*candidate.settings()).into(),
            Self::Cube(candidate) => (*candidate.settings()).into(),
        }
    }

    /// The norm of `input`'s rows scaled by `weight`, written into `out` as a product reads its
    /// lhs ([`RmsNormRoutineCandidate::launch_into`]).
    ///
    /// # Errors
    ///
    /// Why the routine cannot write the norm into `out` at its setting — the plane routine, whose
    /// planes no cube holds, never encodes — or why `out` cannot hold it.
    pub fn launch_into(&self, call: &RmsNormCall, out: LhsOutput) -> Result<(), Decline> {
        match self {
            Self::CubeGroups(candidate) => candidate.launch_into(call, out),
            Self::Plane(candidate) => candidate.launch_into(call, out),
            Self::Cube(candidate) => candidate.launch_into(call, out),
        }
    }

    /// The norm of `input`'s rows scaled by `weight` prepared for a sink
    /// ([`RmsNormRoutineCandidate::sink_launch`]), and given to `visitor`.
    pub fn sink_launch<V: SinkLaunchVisitor>(&self, call: &RmsNormCall, visitor: V) -> V::Output {
        match self {
            Self::CubeGroups(candidate) => visitor.visit(candidate.sink_launch(call)),
            Self::Plane(candidate) => visitor.visit(candidate.sink_launch(call)),
            Self::Cube(candidate) => visitor.visit(candidate.sink_launch(call)),
        }
    }
}

/// The candidate's name: its setting's.
impl core::fmt::Display for RmsNormCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings().fmt(f)
    }
}

impl From<RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine>> for RmsNormCandidate {
    fn from(candidate: RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine>) -> Self {
        Self::CubeGroups(candidate)
    }
}

impl From<RmsNormRoutineCandidate<PlaneRmsNormRoutine>> for RmsNormCandidate {
    fn from(candidate: RmsNormRoutineCandidate<PlaneRmsNormRoutine>) -> Self {
        Self::Plane(candidate)
    }
}

impl From<RmsNormRoutineCandidate<CubeRmsNormRoutine>> for RmsNormCandidate {
    fn from(candidate: RmsNormRoutineCandidate<CubeRmsNormRoutine>) -> Self {
        Self::Cube(candidate)
    }
}

/// The routine's own settings' name: what a candidate is named by, and so what a persisted winner
/// is checksummed by.
impl core::fmt::Display for RmsNormRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeGroups(settings) => settings.fmt(f),
            Self::Plane(settings) => settings.fmt(f),
            Self::Cube(settings) => settings.fmt(f),
        }
    }
}

impl From<CubeGroupsRmsNormRoutineSettings> for RmsNormRoutineSettings {
    fn from(settings: CubeGroupsRmsNormRoutineSettings) -> Self {
        Self::CubeGroups(settings)
    }
}

impl From<PlaneRmsNormRoutineSettings> for RmsNormRoutineSettings {
    fn from(settings: PlaneRmsNormRoutineSettings) -> Self {
        Self::Plane(settings)
    }
}

impl From<CubeRmsNormRoutineSettings> for RmsNormRoutineSettings {
    fn from(settings: CubeRmsNormRoutineSettings) -> Self {
        Self::Cube(settings)
    }
}
