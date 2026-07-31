use cubecl::{
    prelude::*,
    quant::scheme::QuantScheme,
    zspace::{Strides, strides},
};

use crate::InvalidConfigError;

#[derive(CubeType, Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
/// Layout of a 2D structure such as a tensor, shared memory or slice,
/// used within any matmul kernel level
pub enum MatrixLayout {
    #[default]
    RowMajor,
    ColMajor,
}

impl MatrixLayout {
    pub fn from_shape_and_strides(
        shape: &[usize],
        strides: &[usize],
        scheme: Option<&QuantScheme>,
    ) -> Result<Self, InvalidConfigError> {
        assert!(
            shape.len() >= 2 && shape.len() == strides.len(),
            "Shape/stride mismatch or not a matrix"
        );

        if let Some(packing_dim) = scheme.and_then(|s| s.packing_dim()) {
            if packing_dim == 0 {
                return Ok(MatrixLayout::RowMajor);
            }
            if packing_dim == 1 {
                return Ok(MatrixLayout::ColMajor);
            }

            return Err(Box::new(format!(
                "Invalid or non-contiguous matrix layout: packing_dim={packing_dim:?}"
            )));
        }

        let n = shape.len();

        let outer = shape[n - 2];
        let inner = shape[n - 1];

        let stride_outer = strides[n - 2];
        let stride_inner = strides[n - 1];

        // These checks are actually broken for quantized inputs (and are not trivially fixable).
        // For quantized tensors the quantized axis will probably need to be stored, since it can be
        // hard to tell on which axis it is packed.
        // The packed axis is always the contiguous one. One test case has a logical shape of [4, 4]
        // for example, with strides of [1, 1]. It is not possible to determine the packing dimension
        // accurately for this problem.

        // A dimension of size 1 is only ever indexed at 0, so its stride is never
        // added to an offset and cannot make the matrix non-contiguous. Reading it
        // as evidence of the opposite layout would disagree with
        // `matrix_batch_layout`, the classifier the matmul autotune key is built
        // from: a `[1, k]` row vector carrying strides `[1, 1]` is contiguous there
        // but would land here as col major, so a plan tuned for one layout gets
        // replayed on the other.

        // Row-major: inner dimension is contiguous
        if (stride_inner == 1) && (outer == 1 || stride_outer >= inner) {
            return Ok(MatrixLayout::RowMajor);
        }

        // Col-major: outer dimension is contiguous
        if (stride_outer == 1) && (inner == 1 || stride_inner >= outer) {
            return Ok(MatrixLayout::ColMajor);
        }

        Err(Box::new(format!(
            "Invalid or non-contiguous matrix layout: shape={shape:?}, strides={strides:?}",
        )))
    }

    pub fn to_strides(&self, shape: &[usize]) -> Strides {
        assert!(shape.len() >= 2, "Shape must have at least 2 dimensions");

        let n = shape.len();
        let mut strides = strides![0; n];

        // Start with contiguous layout for last two dims
        match self {
            MatrixLayout::RowMajor => {
                strides[n - 1] = 1; // inner dim contiguous
                strides[n - 2] = shape[n - 1]; // outer stride = inner size
            }
            MatrixLayout::ColMajor => {
                strides[n - 2] = 1; // outer dim contiguous
                strides[n - 1] = shape[n - 2]; // inner stride = outer size
            }
        }

        // Batch dims: contiguous
        for i in (0..n - 2).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }
}

#[cfg(feature = "testing")]
impl From<MatrixLayout> for cubek_test_utils::StridedLayout {
    fn from(layout: MatrixLayout) -> Self {
        match layout {
            MatrixLayout::RowMajor => Self::RowMajor,
            MatrixLayout::ColMajor => Self::ColMajor,
        }
    }
}

#[cfg(feature = "testing")]
impl From<MatrixLayout> for cubek_test_utils::LayoutSpec {
    fn from(layout: MatrixLayout) -> Self {
        cubek_test_utils::StridedLayout::from(layout).into()
    }
}

#[cube]
/// Maps the matmul MatrixLayout to cmma's MatrixLayout, for use in Cmma API.
pub fn as_cmma_layout(#[comptime] layout: MatrixLayout) -> cmma::MatrixLayout {
    match layout {
        MatrixLayout::RowMajor => cmma::MatrixLayout::RowMajor,
        MatrixLayout::ColMajor => cmma::MatrixLayout::ColMajor,
    }
}

#[cube]
/// Maps the cmma's MatrixLayout to matmul MatrixLayout.
pub fn from_cmma_layout(#[comptime] layout: cmma::MatrixLayout) -> comptime_type!(MatrixLayout) {
    match layout {
        cmma::MatrixLayout::RowMajor => MatrixLayout::RowMajor,
        cmma::MatrixLayout::ColMajor => MatrixLayout::ColMajor,
        cmma::MatrixLayout::Undefined => MatrixLayout::RowMajor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(shape: &[usize], strides: &[usize]) -> Result<MatrixLayout, String> {
        MatrixLayout::from_shape_and_strides(shape, strides, None).map_err(|err| err.to_string())
    }

    #[test]
    fn contiguous_is_row_major() {
        assert_eq!(layout(&[4, 8], &[8, 1]).unwrap(), MatrixLayout::RowMajor);
    }

    #[test]
    fn transposed_is_col_major() {
        assert_eq!(layout(&[4, 8], &[1, 4]).unwrap(), MatrixLayout::ColMajor);
    }

    #[test]
    fn pitched_rows_are_row_major() {
        // A padded row stride is still row major: rows never overlap.
        assert_eq!(layout(&[4, 8], &[16, 1]).unwrap(), MatrixLayout::RowMajor);
    }

    #[test]
    fn single_row_is_row_major() {
        // `[k, 1]` transposed: the row stride is 1 because there is only ever one
        // row, which must not be read as col major.
        assert_eq!(layout(&[1, 8], &[1, 1]).unwrap(), MatrixLayout::RowMajor);
    }

    #[test]
    fn single_column_is_col_major() {
        // A column of a col-major matrix: the inner stride is below the row count
        // only because the single column is never advanced past.
        assert_eq!(layout(&[8, 1], &[1, 4]).unwrap(), MatrixLayout::ColMajor);
    }

    #[test]
    fn overlapping_strides_are_rejected() {
        assert!(layout(&[4, 8], &[2, 1]).is_err());
    }

    #[test]
    fn batches_do_not_change_the_matrix_layout() {
        assert_eq!(
            layout(&[2, 4, 8], &[32, 8, 1]).unwrap(),
            MatrixLayout::RowMajor
        );
        assert_eq!(
            layout(&[2, 4, 8], &[32, 1, 4]).unwrap(),
            MatrixLayout::ColMajor
        );
    }
}
