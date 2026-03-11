use cubecl::{CubeCount, CubeDim, VectorizationError, ir::StorageType, server::LaunchError};
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

    /// The required CMMA instruction is not supported for the given element types and tile size.
    CmmaInstructionUnavailable {
        lhs: StorageType,
        rhs: StorageType,
        output: StorageType,
    },
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
            AttentionAvailabilityError::CmmaInstructionUnavailable { lhs, rhs, output } => {
                writeln!(
                    f,
                    "Cmma on inputs lhs {lhs:?} rhs {rhs:?} and output {output:?} not supported.",
                )
            }
        }
    }
}

/// Error that arises from invalid configurations
pub type InvalidConfigError = Box<dyn Display>;

/// Error that arises from invalid configurations
pub struct FormattedConfigError {
    func: Box<dyn Fn() -> String>,
}

impl FormattedConfigError {
    #[allow(clippy::new_ret_no_self)]
    pub fn new<F: Fn() -> String + 'static>(func: F) -> Box<dyn Display> {
        Box::new(Self {
            func: Box::new(func),
        })
    }
}

impl Display for FormattedConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = (self.func)();
        write!(f, "{string}")
    }
}
