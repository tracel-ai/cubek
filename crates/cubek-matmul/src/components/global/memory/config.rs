use std::{fmt::Debug, hash::Hash};

use cubecl::ir::StorageType;

use crate::{
    components::global::memory::{GlobalLayoutConfig, ViewDirection},
    definition::MatrixLayout,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct GlobalMemoryConfig {
    pub line_size: u32,
    pub check_row_bounds: bool,
    pub check_col_bounds: bool,
    pub matrix_layout: MatrixLayout,
    pub view_direction: ViewDirection,
    pub dtype: StorageType,
}

impl GlobalMemoryConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        line_size: u32,
        check_row_bounds: bool,
        check_col_bounds: bool,
        matrix_layout: MatrixLayout,
        view_direction: ViewDirection,
        dtype: StorageType,
    ) -> Self {
        GlobalMemoryConfig {
            line_size,
            check_row_bounds,
            check_col_bounds,
            matrix_layout,
            view_direction,
            dtype,
        }
    }

    pub fn as_global_layout_config(self) -> GlobalLayoutConfig {
        GlobalLayoutConfig {
            matrix_layout: self.matrix_layout,
            check_row_bounds: self.check_row_bounds,
            check_col_bounds: self.check_col_bounds,
        }
    }
}
