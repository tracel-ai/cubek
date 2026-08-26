use core::fmt::Debug;
use cubecl::server::LaunchError;
use cubek_matmul::definition::{MatmulAvailabilityError, MatmulSetupError};
use cubek_std::InvalidConfigError;

#[allow(clippy::large_enum_variant)]
pub enum ConvSetupError {
    Matmul(MatmulSetupError),
    Groups(usize),
    /// The depthwise routine was handed a convolution that is not depthwise: it requires
    /// `groups == in_channels == out_channels`, one filter per channel.
    NotDepthwise {
        groups: usize,
        input_channels: usize,
        output_channels: usize,
        weight_channels: usize,
        weight_group_channels: usize,
    },
    /// A caller-provided convolution strategy cannot form valid launch geometry.
    InvalidConfig(InvalidConfigError),
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
            ConvSetupError::NotDepthwise {
                groups,
                input_channels,
                output_channels,
                weight_channels,
                weight_group_channels,
            } => writeln!(
                f,
                "Unable to launch the depthwise convolution because it needs one filter per \
                 channel, but groups is {groups}, input channels is {input_channels}, output \
                 channels is {output_channels}, weight channels is {weight_channels}, and weight \
                 channels per group is {weight_group_channels}",
            ),
            ConvSetupError::InvalidConfig(err) => {
                write!(f, "Invalid convolution config: {err}")
            }
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
