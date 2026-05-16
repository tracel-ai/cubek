use cubek_std::MatrixLayout;

use crate::components::{
    batch::{BatchConfig, CheckBounds},
    global::memory::GlobalLayoutConfig,
};

/// Plane-parallel GEMM is Row-Col only: lhs is RowMajor (K-contiguous) and
/// rhs is ColMajor (K-contiguous). The launch layer enforces this; the only
/// other supported layouts are when an operand is a vector (m=1 or n=1), in
/// which case the matrix-side layout is the only meaningful one and the
/// vector side is contiguous along K by convention.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GemmPlaneParallelConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) check_bounds: CheckBounds,
}

impl BatchConfig for GemmPlaneParallelConfig {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: false,
        }
    }
}
