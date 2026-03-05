use cubecl::{
    LineSizeError, Runtime,
    client::ComputeClient,
    ir::LineSize,
    tensor_line_size_parallel,
    zspace::{Shape, Strides},
};
use cubek_std::MatrixLayout;

use std::fmt::Debug;

use crate::definition::error::MatmulSetupError;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
/// Line size used for each tensor in global memory accesses.
/// Represents the number of elements processed per SIMD load/store.
pub struct MatmulLineSizes {
    pub lhs: LineSize,
    pub rhs: LineSize,
    pub out: LineSize,
}

#[derive(Clone, Debug)]
/// Candidate line sizes supported for each tensor.
///
/// These lists begin with compiler-supported sizes and are progressively
/// filtered based on problem shape divisibility and hardware constraints.
pub struct AvailableLineSizes {
    pub lhs: Vec<LineSize>,
    pub rhs: Vec<LineSize>,
    pub out: Vec<LineSize>,
}

impl AvailableLineSizes {
    pub fn from_type_size_tma<R: Runtime>(client: &ComputeClient<R>, elem_out: usize) -> Self {
        // TMA requires line size 1 for inputs
        AvailableLineSizes {
            lhs: vec![1],
            rhs: vec![1],
            out: client.io_optimized_line_sizes(elem_out).collect(),
        }
    }

    pub fn from_type_sizes<R: Runtime>(
        client: &ComputeClient<R>,
        elem_lhs: usize,
        elem_rhs: usize,
        elem_out: usize,
    ) -> Self {
        AvailableLineSizes {
            lhs: client.io_optimized_line_sizes(elem_lhs).collect(),
            rhs: client.io_optimized_line_sizes(elem_rhs).collect(),
            out: client.io_optimized_line_sizes(elem_out).collect(),
        }
    }

    /// Filter available line sizes considering tensor shapes and strides for Lhs
    pub fn filter_lhs_with_tensor(
        self,
        strides: &Strides,
        shape: &Shape,
        layout: MatrixLayout,
    ) -> Self {
        let rank = strides.len();

        let target = tensor_line_size_parallel(
            self.lhs.iter().copied(),
            shape,
            strides,
            match layout {
                MatrixLayout::RowMajor => rank - 1,
                MatrixLayout::ColMajor => rank - 2,
            },
        );

        self.filter_lhs(move |x| *x == target)
    }

    /// Filter available line sizes considering tensor shapes and strides for Rhs
    pub fn filter_rhs_with_tensor(
        self,
        strides: &Strides,
        shape: &Shape,
        layout: MatrixLayout,
    ) -> Self {
        let rank = strides.len();

        let target = tensor_line_size_parallel(
            self.rhs.iter().copied(),
            shape,
            strides,
            match layout {
                MatrixLayout::RowMajor => rank - 1,
                MatrixLayout::ColMajor => rank - 2,
            },
        );

        self.filter_rhs(move |x| *x == target)
    }

    /// Filter available line sizes considering tensor shapes and strides for output
    pub fn filter_out_with_tensor(self, strides: &Strides, shape: &Shape) -> Self {
        let rank = strides.len();

        let target = tensor_line_size_parallel(self.out.iter().copied(), shape, strides, rank - 1);

        self.filter_out(move |x| *x == target)
    }

    /// Filter available line sizes for Lhs
    pub fn filter_lhs<F>(self, pred: F) -> Self
    where
        F: FnMut(&usize) -> bool,
    {
        Self {
            lhs: self.lhs.iter().copied().filter(pred).collect(),
            rhs: self.rhs,
            out: self.out,
        }
    }

    /// Filter available line sizes for Rhs
    pub fn filter_rhs<F>(self, pred: F) -> Self
    where
        F: FnMut(&usize) -> bool,
    {
        Self {
            lhs: self.lhs,
            rhs: self.rhs.iter().copied().filter(pred).collect(),
            out: self.out,
        }
    }

    /// Filter available line sizes for output
    pub fn filter_out<F>(self, pred: F) -> Self
    where
        F: FnMut(&usize) -> bool,
    {
        Self {
            lhs: self.lhs,
            rhs: self.rhs,
            out: self.out.iter().copied().filter(pred).collect(),
        }
    }

    /// Pick the largest remaining line size for each tensor
    pub fn pick_max(self) -> Result<MatmulLineSizes, MatmulSetupError> {
        let pick = |v: Vec<usize>| {
            v.into_iter()
                .max()
                .ok_or(MatmulSetupError::LineSize(LineSizeError::NoValidLineSize))
        };

        Ok(MatmulLineSizes {
            lhs: pick(self.lhs)?,
            rhs: pick(self.rhs)?,
            out: pick(self.out)?,
        })
    }
}
