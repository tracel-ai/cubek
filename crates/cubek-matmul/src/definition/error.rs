use cubecl::{CubeCount, CubeDim, VectorizationError, ir::ElemType, server::LaunchError};
use cubek_std::{InvalidConfigError, MatrixLayout};
use std::fmt::{Debug, Display};

/// Errors that can occur during the setup phase of a matmul operation.
#[non_exhaustive]
pub enum MatmulSetupError {
    /// A required hardware or runtime feature is not available.
    Unavailable(MatmulAvailabilityError),

    /// The provided configuration is invalid or rejected by a component.
    InvalidConfig(InvalidConfigError),

    /// No compatible vector size could be found for the given constraints.
    Vectorization(VectorizationError),

    /// An error happened during launch.
    Launch(LaunchError),

    /// A launch found false what its routine prepared: a bug in the routine, never a feature the
    /// device lacks nor a problem the routine refuses.
    #[cfg(feature = "tiled")]
    Broken(cubecl::tile::launch::Decline),
}

impl From<LaunchError> for MatmulSetupError {
    fn from(value: LaunchError) -> Self {
        Self::Launch(value)
    }
}

/// A specific feature required for matmul is not available in the current runtime or hardware.
pub enum MatmulAvailabilityError {
    /// The requested cube count exceeds what the runtime or hardware supports.
    CubeCountTooBig(CubeCount),

    /// The requested cube dimensions are too large for the current runtime or hardware.
    CubeDimTooBig(CubeDim),

    /// The requested plane dimension is not supported.
    PlaneDimUnsupported { plane_dim: u32 },

    /// The required data types for input or output are not supported.
    TypesUnavailable {
        lhs: ElemType,
        rhs: ElemType,
        output: ElemType,
    },

    /// The required CMMA instruction is not supported for the given element types and shape.
    CmmaInstructionUnavailable {
        lhs: ElemType,
        rhs: ElemType,
        output: ElemType,
        /// The instruction shape, as `(m, n, k)`.
        size: Option<(u32, u32, u32)>,
    },

    /// Impossible to find a supported tile size for the problem.
    TileSizeNotFound,

    /// The layout of the matmul is unsupported
    LayoutUnsupported {
        lhs: MatrixLayout,
        rhs: MatrixLayout,
    },

    /// Barrier synchronization is not available in the runtime.
    BarrierUnavailable,

    /// TMA (Tensor Memory Access) is not available in the runtime.
    TmaUnavailable,

    /// Async copy (`cp.async`) is not available in the runtime.
    AsyncCopyUnavailable,

    /// Reinterpreting memory is unsupported in the current runtime.
    ReinterpretMemoryUnavailable,

    /// Plane operations like plane_sum are unavailable
    PlaneOpsUnavailable,

    /// The kernel's shared-memory footprint exceeds the device's per-cube limit.
    SharedMemoryTooBig { requested: usize, available: usize },
}
impl From<MatmulAvailabilityError> for MatmulSetupError {
    fn from(value: MatmulAvailabilityError) -> Self {
        Self::Unavailable(value)
    }
}

impl From<InvalidConfigError> for MatmulSetupError {
    fn from(value: InvalidConfigError) -> Self {
        Self::InvalidConfig(value)
    }
}

impl From<VectorizationError> for MatmulSetupError {
    fn from(value: VectorizationError) -> Self {
        Self::Vectorization(value)
    }
}

/// A launch the tile engine refused past what the device holds is a feature the device lacks, so
/// a caller falls back as it would without the feature; a [`Broken`](cubecl::tile::launch::Decline::Broken) decline is
/// [`MatmulSetupError::Broken`], kept as it is; any other decline is a configuration the routine
/// cannot run.
#[cfg(feature = "tiled")]
impl From<cubecl::tile::launch::Decline> for MatmulSetupError {
    fn from(decline: cubecl::tile::launch::Decline) -> Self {
        use cubecl::tile::launch::Decline;
        match decline {
            Decline::CubePastDevice { units, .. } => {
                MatmulAvailabilityError::CubeDimTooBig(CubeDim::new_1d(units)).into()
            }
            Decline::CubeShapePastDevice {
                shape: (x, y, z), ..
            } => MatmulAvailabilityError::CubeDimTooBig(CubeDim::new_3d(x, y, z)).into(),
            Decline::GridPastDevice {
                count: (x, y, z), ..
            } => MatmulAvailabilityError::CubeCountTooBig(CubeCount::new_3d(x, y, z)).into(),
            Decline::SharedMemoryPastDevice { bytes, most } => {
                MatmulAvailabilityError::SharedMemoryTooBig {
                    requested: bytes,
                    available: most,
                }
                .into()
            }
            Decline::PlanesUnavailable => MatmulAvailabilityError::PlaneOpsUnavailable.into(),
            decline if decline.is_broken() => Self::Broken(decline),
            decline => Self::InvalidConfig(Box::new(decline)),
        }
    }
}

impl Display for MatmulSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Debug for MatmulSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulSetupError::Unavailable(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because a required feature is unavailable: {err:?}"
                )
            }
            MatmulSetupError::InvalidConfig(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because the config is invalid: {:?}",
                    err.to_string()
                )
            }
            MatmulSetupError::Vectorization(err) => {
                writeln!(
                    f,
                    "Unable to launch matmul because could not find supported vector size: {err:?}"
                )
            }
            MatmulSetupError::Launch(err) => {
                writeln!(f, "Unable to launch matmul with err: {err:?}")
            }
            #[cfg(feature = "tiled")]
            MatmulSetupError::Broken(decline) => {
                writeln!(f, "Unable to launch matmul because a routine is {decline}")
            }
        }
    }
}

impl Debug for MatmulAvailabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulAvailabilityError::CubeCountTooBig(count) => {
                writeln!(f, "Cube count too big {count:?}")
            }
            MatmulAvailabilityError::CubeDimTooBig(dim) => {
                writeln!(f, "Cube dim too big {dim:?}")
            }
            MatmulAvailabilityError::PlaneDimUnsupported { plane_dim } => {
                writeln!(
                    f,
                    "Plane dimension unsupported: {plane_dim}. Only 32 & 64 are supported."
                )
            }
            MatmulAvailabilityError::TypesUnavailable { lhs, rhs, output } => {
                writeln!(
                    f,
                    "Types lhs={lhs:?}, rhs={rhs:?} and/or output={output:?} not supported.",
                )
            }
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                lhs,
                rhs,
                output,
                size: Some((m, n, k)),
            } => writeln!(
                f,
                "Cmma on lhs {lhs:?} rhs {rhs:?} and output {output:?} with shape m={m:?}, n={n:?}, k={k:?} not supported.",
            ),
            MatmulAvailabilityError::LayoutUnsupported { lhs, rhs } => {
                writeln!(
                    f,
                    "Cmma with layouts lhs {lhs:?} and rhs {rhs:?} not supported."
                )
            }
            MatmulAvailabilityError::CmmaInstructionUnavailable {
                lhs,
                rhs,
                output,
                size: None,
            } => writeln!(
                f,
                "Cmma on inputs lhs {lhs:?} rhs {rhs:?} and output {output:?} not supported.",
            ),
            MatmulAvailabilityError::BarrierUnavailable => {
                writeln!(f, "Barrier is not available.")
            }
            MatmulAvailabilityError::TmaUnavailable => {
                writeln!(f, "TMA is not available.")
            }
            MatmulAvailabilityError::AsyncCopyUnavailable => {
                writeln!(f, "Async copy is not available.")
            }
            MatmulAvailabilityError::ReinterpretMemoryUnavailable => {
                writeln!(f, "Memory reinterpretation is not available.")
            }
            MatmulAvailabilityError::PlaneOpsUnavailable => {
                writeln!(f, "Plane-wide operations like plane_sum are not available.")
            }
            MatmulAvailabilityError::TileSizeNotFound => {
                writeln!(f, "No tile size is available for the problem.")
            }
            MatmulAvailabilityError::SharedMemoryTooBig {
                requested,
                available,
            } => writeln!(
                f,
                "Shared memory budget exceeded: kernel requests {requested} bytes, hardware allows {available} bytes per cube."
            ),
        }
    }
}
