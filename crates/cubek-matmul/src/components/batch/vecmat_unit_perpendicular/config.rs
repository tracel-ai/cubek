use cubek_std::MatrixLayout;

use crate::components::{batch::BatchConfig, global::memory::GlobalLayoutConfig};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct VecMatUnitPerpendicularConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
}

impl BatchConfig for VecMatUnitPerpendicularConfig {
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
