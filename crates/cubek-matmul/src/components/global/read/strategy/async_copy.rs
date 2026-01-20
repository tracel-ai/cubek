use cubecl::prelude::barrier::{copy_async, copy_async_checked};
use cubecl::prelude::*;
use cubecl::std::{
    tensor::{View, layout::Coords2d},
    type_size,
};

use crate::components::{
    global::GlobalReaderConfig,
    stage::{StridedStageMemory, TilingLayout},
};
use crate::definition::MatrixLayout;

/// The instruction has a max width of 128 bits, even on Blackwell which supports 256-bit loads
pub(crate) const ASYNC_COPY_WIDTH: u32 = 128;

#[cube]
pub(crate) fn async_copy_from<EG: CubePrimitive, ES: Numeric, T: TilingLayout>(
    view: View<Line<EG>, Coords2d>,
    pos: Coords2d,
    stage: &mut StridedStageMemory<ES, T>,
    stage_offset: u32,
    #[comptime] config: GlobalReaderConfig,
    #[comptime] copy_line_size: u32,
) {
    let mut stage_slice = stage.as_slice_mut(stage.smem.line_size());
    let slice_size = match config.smem_config.matrix_layout {
        MatrixLayout::RowMajor => (1u32, copy_line_size),
        MatrixLayout::ColMajor => (copy_line_size, 1u32),
    }
    .runtime();

    let mut slice_len_global = copy_line_size.runtime();
    let slice_len_stage = copy_line_size / stage_slice.line_size() as u32;

    if config.gmem_config.check_row_bounds {
        let pos = pos.0;
        let shape = view.shape().0;
        match config.gmem_config.matrix_layout {
            MatrixLayout::RowMajor => {
                slice_len_global *= u32::cast_from(pos < shape);
            }
            MatrixLayout::ColMajor => {
                slice_len_global = shape.saturating_sub(pos).min(slice_len_global);
            }
        }
    }

    if config.gmem_config.check_col_bounds {
        let pos = pos.1;
        let shape = view.shape().1;
        match config.gmem_config.matrix_layout {
            MatrixLayout::RowMajor => {
                slice_len_global = shape.saturating_sub(pos).min(slice_len_global);
            }
            MatrixLayout::ColMajor => {
                slice_len_global *= u32::cast_from(pos < shape);
            }
        }
    }

    slice_len_global /= view.line_size() as u32;

    let global_slice = view.slice_unchecked(pos, slice_size).to_linear_slice();

    let type_size = type_size::<ES>(stage_slice.line_size());
    let offset = stage.swizzle.apply(stage_offset, type_size);

    let stage_slice = stage_slice.slice_mut(offset as usize, (offset + slice_len_stage) as usize);

    if config.gmem_config.check_row_bounds || config.gmem_config.check_col_bounds {
        copy_async_checked(
            &global_slice.slice(0, slice_len_global as usize),
            &mut stage_slice.downcast(),
            copy_line_size,
        );
    } else {
        copy_async(&global_slice, &mut stage_slice.downcast(), copy_line_size);
    }
}
