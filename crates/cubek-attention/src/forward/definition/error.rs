use cubecl::{CubeCount, CubeDim, VectorizationError, server::LaunchError};
use cubek_matmul::definition::MatmulAvailabilityError;
use cubek_std::InvalidConfigError;
use std::fmt::{Debug, Display};

/// Errors that can occur during the setup phase of an attention operation.
pub enum AttentionSetupError {
    /// A required hardware or runtime feature is not available.
    Unavailable(AttentionAvailabilityError),

    /// The provided configuration is invalid or rejected by a component.
    InvalidConfig(InvalidConfigError),

    /// No compatible vector size could be found for the given constraints.
    Vectorization(VectorizationError),

    /// An error that happened during execution.
    Execution(LaunchError),
}

/// A specific feature required for attention is not available in the current runtime or hardware.
pub enum AttentionAvailabilityError {
    /// The requested cube count exceeds what the runtime or hardware supports.
    CubeCountTooBig(CubeCount),

    /// The requested cube dimensions are too large for the current runtime or hardware.
    CubeDimTooBig(CubeDim),

    /// The required matmul instruction is not supported for the given element types and tile size.
    MatmulInstructionUnavailable(MatmulAvailabilityError),

    /// Plane (warp/subgroup) operations are required but not available on this device.
    PlaneOpsUnavailable,
}

impl From<AttentionAvailabilityError> for AttentionSetupError {
    fn from(value: AttentionAvailabilityError) -> Self {
        Self::Unavailable(value)
    }
}

impl From<InvalidConfigError> for AttentionSetupError {
    fn from(value: InvalidConfigError) -> Self {
        Self::InvalidConfig(value)
    }
}

impl From<VectorizationError> for AttentionSetupError {
    fn from(value: VectorizationError) -> Self {
        Self::Vectorization(value)
    }
}

impl Display for AttentionSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Debug for AttentionSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionSetupError::Unavailable(err) => {
                writeln!(
                    f,
                    "Unable to launch attention because a required feature is unavailable: {err:?}"
                )
            }
            AttentionSetupError::InvalidConfig(err) => {
                writeln!(
                    f,
                    "Unable to launch attention because the config is invalid: {:?}",
                    err.to_string()
                )
            }
            AttentionSetupError::Vectorization(err) => {
                writeln!(
                    f,
                    "Unable to launch attention because could not find supported vectorization: {err:?}"
                )
            }
            AttentionSetupError::Execution(error) => {
                writeln!(f, "{error:?}")
            }
        }
    }
}

impl Debug for AttentionAvailabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionAvailabilityError::CubeCountTooBig(count) => {
                writeln!(f, "Cube count too big {count:?}")
            }
            AttentionAvailabilityError::CubeDimTooBig(dim) => {
                writeln!(f, "Cube dim too big {dim:?}")
            }
            AttentionAvailabilityError::MatmulInstructionUnavailable(error) => {
                writeln!(f, "Matmul is not supported: {error:?}",)
            }
            AttentionAvailabilityError::PlaneOpsUnavailable => {
                writeln!(f, "Plane operations are not supported on this device")
            }
        }
    }
}
