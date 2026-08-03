use cubecl::ir::StorageType;

use crate::definition::QRSetupError;

/// Runtime description of a QR factorization problem: an `m x n` matrix with `m >= n`.
///
/// Only runtime information belongs here (shapes, element type); anything that
/// changes the generated kernel code goes in a routine blueprint instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QRProblem {
    /// Number of rows (`m`) of the input matrix.
    pub rows: usize,
    /// Number of columns (`n`) of the input matrix.
    pub cols: usize,
    /// Element type of the input matrix.
    pub dtype: StorageType,
}

impl QRProblem {
    /// Build a problem description from a tensor shape, validating that it
    /// describes a supported (`m >= n`) matrix.
    pub fn from_shape(shape: &[usize], dtype: StorageType) -> Result<Self, QRSetupError> {
        if shape.len() != 2 || shape[0] < shape[1] || shape[1] == 0 {
            return Err(QRSetupError::InvalidShape);
        }
        Ok(Self {
            rows: shape[0],
            cols: shape[1],
            dtype,
        })
    }
}
