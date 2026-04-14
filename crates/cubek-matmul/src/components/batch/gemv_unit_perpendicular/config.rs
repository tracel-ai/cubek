use cubek_std::MatrixLayout;

use crate::components::{
    batch::{BatchConfig, CheckBounds},
    global::memory::GlobalLayoutConfig,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct VecMatUnitPerpendicularConfig {
    pub(crate) plane_dim: u32,
    pub(crate) num_planes: u32,
    pub(crate) check_bounds: CheckBounds,
}

impl BatchConfig for VecMatUnitPerpendicularConfig {
    fn lhs_global_layout_config(&self) -> GlobalLayoutConfig {
        let checked = self.check_bounds == CheckBounds::Checked;
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: checked,
        }
    }

    fn rhs_global_layout_config(&self) -> GlobalLayoutConfig {
        let checked = self.check_bounds == CheckBounds::Checked;
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::ColMajor,
            check_row_bounds: checked,
            check_col_bounds: checked,
        }
    }

    fn out_global_layout_config(&self) -> GlobalLayoutConfig {
        let checked = self.check_bounds == CheckBounds::Checked;
        GlobalLayoutConfig {
            matrix_layout: MatrixLayout::RowMajor,
            check_row_bounds: false,
            check_col_bounds: checked,
        }
    }
}
