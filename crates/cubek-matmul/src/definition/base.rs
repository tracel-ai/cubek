use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
/// Description of a matmul problem to solve, regardless of actual data
pub struct MatmulProblem {
    /// Number of rows in the output matrix
    pub m: usize,
    /// Number of columns in the output matrix
    pub n: usize,
    /// Reduction dimension
    pub k: usize,

    /// Batch shape for Lhs tensor
    pub lhs_batches: Vec<usize>,
    /// Batch shape for Rhs tensor
    pub rhs_batches: Vec<usize>,
    /// Batch shape for Out tensor
    pub out_batches: Vec<usize>,

    /// Shape of Lhs tensor
    pub lhs_shape: Vec<usize>,
    /// Shape of Rhs tensor
    pub rhs_shape: Vec<usize>,
    /// Shape of Out tensor
    pub out_shape: Vec<usize>,

    /// Strides for the Lhs tensor
    pub lhs_strides: Vec<usize>,
    /// Strides for the Rhs tensor
    pub rhs_strides: Vec<usize>,
    /// Strides for the Out tensor
    pub out_strides: Vec<usize>,

    /// Memory layout of the Lhs matrix.
    pub lhs_layout: MatrixLayout,
    /// Memory layout of the Rhs matrix.
    pub rhs_layout: MatrixLayout,
    /// Memory layout of the Out matrix.
    pub out_layout: MatrixLayout,
}

impl MatmulProblem {
    pub fn from_shapes_and_strides(
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
        out_shape: Vec<usize>,
        lhs_strides: Vec<usize>,
        rhs_strides: Vec<usize>,
        out_strides: Vec<usize>,
    ) -> Self {
        let rank = out_shape.len();
        let lhs_layout = MatrixLayout::from_shape_and_strides(&lhs_shape, &lhs_strides);
        let rhs_layout = MatrixLayout::from_shape_and_strides(&rhs_shape, &rhs_strides);
        let out_layout = MatrixLayout::from_shape_and_strides(&out_shape, &out_strides);

        Self {
            m: lhs_shape[rank - 2],
            n: rhs_shape[rank - 1],
            k: lhs_shape[rank - 1],
            lhs_batches: lhs_shape[..lhs_shape.len() - 2].to_vec(),
            rhs_batches: rhs_shape[..rhs_shape.len() - 2].to_vec(),
            out_batches: out_shape[..out_shape.len() - 2].to_vec(),
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_strides,
            rhs_strides,
            out_strides,
            lhs_layout,
            rhs_layout,
            out_layout,
        }
    }

    pub fn from_parameters(
        m: usize,
        n: usize,
        k: usize,
        batches: Vec<usize>,
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        out_layout: MatrixLayout,
    ) -> Self {
        let lhs_shape: Vec<usize> = batches.iter().cloned().chain(vec![m, k]).collect();
        let rhs_shape: Vec<usize> = batches.iter().cloned().chain(vec![k, n]).collect();
        let out_shape: Vec<usize> = batches.iter().cloned().chain(vec![m, n]).collect();

        let lhs_strides = lhs_layout.to_strides(&lhs_shape);
        let rhs_strides = rhs_layout.to_strides(&rhs_shape);
        let out_strides = out_layout.to_strides(&out_shape);

        Self {
            m,
            n,
            k,
            lhs_batches: batches.clone(),
            rhs_batches: batches.clone(),
            out_batches: batches,
            lhs_shape,
            rhs_shape,
            out_shape,
            lhs_strides,
            rhs_strides,
            out_strides,
            lhs_layout,
            rhs_layout,
            out_layout,
        }
    }

    /// Returns the total number of batches of the output
    pub fn num_batches(&self) -> usize {
        self.out_batches.iter().product()
    }
}

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize)]
/// Interpretation of matrix multiplication based on input shapes.
pub enum MatmulKind {
    /// (M, K) @ (K, N) → (M, N), with M, K, N > 1
    General,

    /// (M, K) @ (K, 1) → (M, 1)
    MatVec,

    /// (1, K) @ (K, N) → (1, N)
    VecMat,

    /// (1, 1) @ (1, N) → (1, N)
    ScalarVec,

    /// (M, 1) @ (1, 1) → (M, 1)
    VecScalar,

    /// (1, K) @ (K, 1) → (1, 1)
    InnerProduct,

    /// (M, 1) @ (1, N) → (M, N)
    OuterProduct,

    /// (1, 1) @ (1, 1) → (1, 1)
    ScalarProduct,
}

impl From<MatmulProblemSize> for MatmulKind {
    fn from(matmul_size: MatmulProblemSize) -> Self {
        enum DimKind {
            Scalar,
            Vector,
        }

        impl From<u32> for DimKind {
            fn from(x: u32) -> Self {
                match x {
                    1 => DimKind::Scalar,
                    _ => DimKind::Vector,
                }
            }
        }

        use DimKind::*;
        match (
            matmul_size.m().into(),
            matmul_size.n().into(),
            matmul_size.k().into(),
        ) {
            (Scalar, Scalar, Scalar) => MatmulKind::ScalarProduct,
            (Scalar, Scalar, Vector) => MatmulKind::InnerProduct,
            (Scalar, Vector, Scalar) => MatmulKind::ScalarVec,
            (Scalar, Vector, Vector) => MatmulKind::VecMat,
            (Vector, Scalar, Scalar) => MatmulKind::VecScalar,
            (Vector, Scalar, Vector) => MatmulKind::MatVec,
            (Vector, Vector, Scalar) => MatmulKind::OuterProduct,
            (Vector, Vector, Vector) => MatmulKind::General,
        }
    }
}

impl From<MatmulProblem> for MatmulProblemSize {
    fn from(problem: MatmulProblem) -> Self {
        MatmulProblemSize::new(problem.m as u32, problem.n as u32, problem.k as u32)
    }
}

impl From<MatmulProblem> for MatmulKind {
    fn from(problem: MatmulProblem) -> Self {
        MatmulProblemSize::new(problem.m as u32, problem.n as u32, problem.k as u32).into()
    }
}

impl From<&MatmulProblem> for MatmulKind {
    fn from(problem: &MatmulProblem) -> Self {
        MatmulProblemSize::new(problem.m as u32, problem.n as u32, problem.k as u32).into()
    }
}

#[derive(CubeType, Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
/// Layout of a 2D structure such as a tensor, shared memory or slice,
/// used within any matmul kernel level
pub enum MatrixLayout {
    #[default]
    RowMajor,
    ColMajor,
}

impl MatrixLayout {
    pub fn from_shape_and_strides(shape: &[usize], strides: &[usize]) -> Self {
        assert!(
            shape.len() >= 2 && shape.len() == strides.len(),
            "Shape/stride mismatch or not a matrix"
        );

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
        if stride_inner == 1 && stride_outer >= inner {
            return MatrixLayout::RowMajor;
        }

        // Col-major: outer dimension is contiguous
        if stride_outer == 1 && stride_inner >= outer {
            return MatrixLayout::ColMajor;
        }

        panic!(
            "Invalid or non-contiguous matrix layout: shape={:?}, strides={:?}",
            shape, strides
        );
    }

    pub fn to_strides(&self, shape: &[usize]) -> Vec<usize> {
        assert!(shape.len() >= 2, "Shape must have at least 2 dimensions");

        let n = shape.len();
        let mut strides = vec![0; n];

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

use crate::{components::global::memory::ViewDirection, definition::MatmulProblemSize};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for all three tensors in a matmul
///
/// Useful to specialize some functions depending on the tensor
pub enum MatmulIdent {
    Lhs,
    Rhs,
    Out,
}

impl MatmulIdent {
    /// Equivalent to into, but type inference works better within Cube functions
    pub fn into_stage(self) -> StageIdent {
        self.into()
    }

    pub fn view_direction(&self) -> ViewDirection {
        match self {
            MatmulIdent::Lhs => ViewDirection::Col,
            MatmulIdent::Rhs => ViewDirection::Row,
            MatmulIdent::Out => ViewDirection::None,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageIdent {
    Lhs,
    Rhs,
    Acc,
    Out,
}

impl From<MatmulIdent> for StageIdent {
    fn from(matmul_ident: MatmulIdent) -> Self {
        match matmul_ident {
            MatmulIdent::Lhs => StageIdent::Lhs,
            MatmulIdent::Rhs => StageIdent::Rhs,
            MatmulIdent::Out => StageIdent::Acc,
        }
    }
}

impl From<StageIdent> for MatmulIdent {
    fn from(matmul_ident: StageIdent) -> Self {
        match matmul_ident {
            StageIdent::Lhs => MatmulIdent::Lhs,
            StageIdent::Rhs => MatmulIdent::Rhs,
            StageIdent::Acc => MatmulIdent::Out,
            StageIdent::Out => MatmulIdent::Out,
        }
    }
}
