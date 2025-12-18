use cubecl::prelude::barrier::copy_async_checked;
use cubecl::prelude::*;
use cubecl::std::{
    tensor::{View, layout::Coords2d},
    type_size,
};
use cubek_matmul::components::{
    global::GlobalReaderConfig,
    stage::{StridedStageMemory, TilingLayout},
};
use cubek_matmul::definition::{MatrixLayout, StageIdent};

use crate::components::{ConvolutionOperation, global::args::RuntimeArgs};

/// The instruction has a max width of 128 bits, even on Blackwell which supports 256-bit loads
pub(crate) const ASYNC_COPY_WIDTH: u32 = 128;

/// Custom version of async copy to clamp slice on channels, not `k` as a whole.
#[cube]
pub(crate) fn async_copy_from<EG: CubePrimitive, ES: Numeric, T: TilingLayout>(
    view: View<Line<EG>, Coords2d>,
    pos: Coords2d,
    stage: &mut StridedStageMemory<ES, T>,
    stage_offset: u32,
    runtime_args: &RuntimeArgs,
    k_offset: u32,
    #[comptime] config: GlobalReaderConfig,
    #[comptime] copy_line_size: u32,
) {
    let operation = comptime![runtime_args.operation];

    let mut stage_slice = stage.as_slice_mut(stage.smem.line_size());
    let slice_size = comptime![match config.smem_config.matrix_layout {
        MatrixLayout::RowMajor => (1u32, copy_line_size),
        MatrixLayout::ColMajor => (copy_line_size, 1u32),
    }]
    .runtime();

    let mut slice_len_global = copy_line_size.runtime();
    let slice_len_stage = copy_line_size / stage_slice.line_size();

    match (config.stage_ident, operation) {
        (StageIdent::Lhs, ConvolutionOperation::Forward)
        | (StageIdent::Lhs, ConvolutionOperation::ForwardTransposed)
        | (StageIdent::Lhs, ConvolutionOperation::BackwardData)
        | (StageIdent::Rhs, ConvolutionOperation::BackwardWeight) => {
            // im2col can give negative spatial indices so need to do a full bounds check on Lhs
            slice_len_global *= u32::cast_from(view.is_in_bounds(pos));

            if config.gmem_config.check_col_bounds {
                let in_c = runtime_args.padded_channels.modulo(k_offset + pos.1);
                slice_len_global = Min::min(
                    SaturatingSub::saturating_sub(runtime_args.channels, in_c),
                    slice_len_global,
                );
            }
        }
        (StageIdent::Rhs, ConvolutionOperation::Forward)
        | (StageIdent::Out, ConvolutionOperation::BackwardWeight) => {
            if config.gmem_config.check_row_bounds {
                let in_c = runtime_args.padded_channels.modulo(k_offset + pos.0);
                slice_len_global = Min::min(
                    SaturatingSub::saturating_sub(runtime_args.channels, in_c),
                    slice_len_global,
                );
            }
            if config.gmem_config.check_col_bounds {
                slice_len_global *= u32::cast_from(pos.1 < view.shape().1);
            }
        }
        (StageIdent::Rhs, ConvolutionOperation::ForwardTransposed)
        | (StageIdent::Rhs, ConvolutionOperation::BackwardData) => {
            if config.gmem_config.check_row_bounds {
                let out_c = runtime_args.padded_channels.modulo(k_offset + pos.0);
                slice_len_global *=
                    u32::cast_from(out_c < runtime_args.channels && pos.0 < view.shape().0);
            }
            if config.gmem_config.check_col_bounds {
                let pos = pos.1;
                let shape = view.shape().1;
                slice_len_global =
                    Min::min(SaturatingSub::saturating_sub(shape, pos), slice_len_global);
            }
        }
        _ => {
            if config.gmem_config.check_row_bounds {
                let pos = pos.0;
                let shape = view.shape().0;
                match config.gmem_config.matrix_layout {
                    MatrixLayout::RowMajor => {
                        slice_len_global *= u32::cast_from(pos < shape);
                    }
                    MatrixLayout::ColMajor => {
                        slice_len_global =
                            Min::min(SaturatingSub::saturating_sub(shape, pos), slice_len_global);
                    }
                }
            }

            if config.gmem_config.check_col_bounds {
                let pos = pos.1;
                let shape = view.shape().1;
                match config.gmem_config.matrix_layout {
                    MatrixLayout::RowMajor => {
                        slice_len_global =
                            Min::min(SaturatingSub::saturating_sub(shape, pos), slice_len_global);
                    }
                    MatrixLayout::ColMajor => {
                        slice_len_global *= u32::cast_from(pos < shape);
                    }
                }
            }
        }
    }

    slice_len_global /= view.line_size();

    let global_slice = view.slice_unchecked(pos, slice_size).to_linear_slice();

    let type_size = type_size::<ES>(stage_slice.line_size());
    let offset = stage.swizzle.apply(stage_offset, type_size);

    let stage_slice = stage_slice.slice_mut(offset, offset + slice_len_stage);

    copy_async_checked(
        &global_slice.slice(0, slice_len_global),
        &mut stage_slice.try_cast_unchecked(),
        copy_line_size,
    );
}
