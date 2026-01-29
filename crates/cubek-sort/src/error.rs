use cubecl::server::LaunchError;
use thiserror::Error;

/// Errors that can occur during sort operations.
#[derive(Error, Debug)]
pub enum SortError {
    /// The input size is zero.
    #[error("Cannot sort an empty array")]
    EmptyInput,

    /// The input and output tensor sizes don't match.
    #[error("Input size ({input}) does not match output size ({output})")]
    SizeMismatch { input: usize, output: usize },

    /// The number of items exceeds the maximum supported.
    #[error("Number of items ({count}) exceeds maximum supported ({max})")]
    TooManyItems { count: usize, max: usize },

    /// An error occurred during kernel launch.
    #[error("An error happened during launch\nCaused by:\n  {0}")]
    Launch(LaunchError),
}
