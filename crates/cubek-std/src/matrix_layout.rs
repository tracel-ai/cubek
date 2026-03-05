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

        // Row-major: inner dimension is contiguous
        if (stride_inner == 1) && stride_outer >= inner {
            return Ok(MatrixLayout::RowMajor);
        }

        // Col-major: outer dimension is contiguous
        if (stride_outer == 1) && stride_inner >= outer {
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
