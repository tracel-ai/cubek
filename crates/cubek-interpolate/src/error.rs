use thiserror::Error;

#[derive(Error, Debug, Clone)]
/// This error should be caught and properly handled.
pub enum InterpolateError {
    /// Default error for unimplemented modes or other generic errors.
    #[error("An error occurred during interpolation.")]
    DefaultError,
}
