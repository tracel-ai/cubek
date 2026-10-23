use crate::multi_level::{
    components::global::memory::GlobalMemoryConfig, stage::StageMemoryConfig,
};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d},
};
use cubek_std::MatrixLayout;

/// Reads data from the tensor view as a window, i.e. a slice of global memory
///
/// The length of the slice is the width of the tile
///
/// # Note
///
/// If the slice would be partly out-of-bounds, it will simply be shorter.
/// The caller must do the padding if necessary.
#[cube]
pub fn load_window_in_stage<EG: CubePrimitive>(
    view: View<'_, EG, Coords2d>,
    nth_window: u32,
    #[comptime] smem_config: StageMemoryConfig,
    #[comptime] gmem_config: GlobalMemoryConfig,
) -> &[EG] {
    let size = match smem_config.matrix_layout {
        MatrixLayout::RowMajor => (1u32, smem_config.elements_per_stage_along_col()).runtime(),
        MatrixLayout::ColMajor => (smem_config.elements_per_stage_along_row(), 1u32).runtime(),
    };

    load_window(view, nth_window, size, gmem_config)
}

#[cube]
fn load_window<EG: CubePrimitive>(
    view: View<'_, EG, Coords2d>,
    nth_window: u32,
    size: Coords2d,
    #[comptime] gmem_config: GlobalMemoryConfig,
) -> &[EG] {
    let offset = match gmem_config.matrix_layout {
        MatrixLayout::RowMajor => (nth_window, 0),
        MatrixLayout::ColMajor => (0, nth_window),
    };

    if gmem_config.check_row_bounds || gmem_config.check_col_bounds {
        let view = view.slice(offset, size);
        view.as_linear_slice()
    } else {
        let view = view.slice_unchecked(offset, size);
        view.as_linear_slice()
    }
}
