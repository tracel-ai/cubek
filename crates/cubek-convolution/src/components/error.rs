use core::fmt::Debug;
use cubecl::server::LaunchError;
use cubek_matmul::definition::{MatmulAvailabilityError, MatmulSetupError};

#[allow(clippy::large_enum_variant)]
pub enum ConvSetupError {
    Matmul(MatmulSetupError),
    Groups(usize),
    /// The depthwise routine was handed a convolution that is not depthwise: it requires
    /// `groups == in_channels == out_channels`, one filter per channel.
    NotDepthwise {
        groups: usize,
        channels: usize,
    },
    Unknown,
    Launch(LaunchError),
}

impl From<LaunchError> for ConvSetupError {
    fn from(value: LaunchError) -> Self {
        Self::Launch(value)
    }
}

impl Debug for ConvSetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvSetupError::Matmul(err) => {
                write!(f, "{err:?}")
            }
            ConvSetupError::Groups(groups) => {
                writeln!(
                    f,
                    "Unable to launch matmul because groups must be one, is actually {groups}",
                )
            }
            ConvSetupError::NotDepthwise { groups, channels } => writeln!(
                f,
                "Unable to launch the depthwise convolution because it needs one filter per \
                 channel, but groups is {groups} against {channels} channels",
            ),
            ConvSetupError::Unknown => write!(f, "Unknown"),
            ConvSetupError::Launch(err) => write!(f, "Launch error {err:?}"),
        }
    }
}

impl From<MatmulSetupError> for ConvSetupError {
    fn from(value: MatmulSetupError) -> Self {
        Self::Matmul(value)
    }
}

impl From<MatmulAvailabilityError> for ConvSetupError {
    fn from(value: MatmulAvailabilityError) -> Self {
        Self::Matmul(MatmulSetupError::Unavailable(value))
    }
}

#[allow(clippy::from_over_into)]
impl Into<String> for ConvSetupError {
    fn into(self) -> String {
        format!("{self:?}")
    }
}
