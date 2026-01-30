use cubecl::server::LaunchError;
use thiserror::Error;

/// Errors that can occur during sort operations.
#[derive(Error, Debug)]
pub enum SortError {
    /// An error occurred during kernel launch.
    #[error("Kernel launch failed: {0}")]
    Launch(LaunchError),
}
