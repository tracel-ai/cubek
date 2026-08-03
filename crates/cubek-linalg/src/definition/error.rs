use cubecl::ir::StorageType;
use thiserror::Error;

/// Errors that can occur when trying to launch QR decomposition.
#[derive(Debug, Error, PartialEq, Eq, Clone, Hash)]
pub enum QRSetupError {
    /// The input should be a non-empty matrix where m should be greater or equal to n.
    #[error("The input should be a non-empty matrix where m should be greater or equal to n.")]
    InvalidShape,
    /// The element type the kernels were launched with does not match the tensor's.
    #[error("Element type mismatch: launched as {launched:?} but the tensor holds {actual:?}.")]
    TypeMismatch {
        /// Element type the generic entry point was instantiated with.
        launched: StorageType,
        /// Element type of the tensor handle.
        actual: StorageType,
    },
    /// An internal matmul refused the requested launch.
    #[error("Internal matmul setup failed: {0}")]
    Matmul(String),
    /// The routine requires more shared memory than the hardware provides.
    #[error(
        "The routine requires {requested} bytes of shared memory but only {available} are available."
    )]
    SharedMemoryLimitExceeded {
        /// Number of shared memory bytes the routine asked for.
        requested: usize,
        /// Number of shared memory bytes the hardware provides.
        available: usize,
    },
    /// A forced blueprint is inconsistent with the problem or the hardware.
    #[error("Invalid blueprint: {0}")]
    InvalidBlueprint(String),
}
