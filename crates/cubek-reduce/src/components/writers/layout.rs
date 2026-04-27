use cubecl::{
    prelude::*,
    std::tensor::{
        layout::{Coords1d, Coords2d, Layout, LayoutExpand},
        r#virtual::VirtualTensor,
    },
};

use crate::components::args::NumericVector;

/// Maps a `(write_index, k_iter)` coordinate to a flat vector position in the
/// output buffer. Strides are expressed in vector units (one step along the
/// output's SIMD axis = one unit in `write_stride`).
///
/// For rank-1 outputs (or any case where `reduce_axis == out_vec_axis`), the
/// caller should pass `write_stride = 0` and `num_writes = 1`, so the layout
/// collapses to `position = k_iter * k_stride`.
#[derive(CubeType, Clone)]
pub struct ReduceOutputLayout {
    k_stride: usize,
    write_stride: usize,
    num_writes: usize,
    accumulator_length: usize,
}

#[cube]
impl ReduceOutputLayout {
    pub fn new(
        k_stride: usize,
        write_stride: usize,
        num_writes: usize,
        accumulator_length: usize,
    ) -> ReduceOutputLayout {
        ReduceOutputLayout {
            k_stride,
            write_stride,
            num_writes,
            accumulator_length,
        }
    }
}

#[cube]
impl Layout for ReduceOutputLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, coords: Self::Coordinates) -> Coords1d {
        let write_index = coords.0 as usize;
        let k_iter = coords.1 as usize;
        k_iter * self.k_stride + write_index * self.write_stride
    }

    fn to_source_pos_checked(&self, coords: Self::Coordinates) -> (Coords1d, bool) {
        (self.to_source_pos(coords), self.is_in_bounds(coords))
    }

    fn shape(&self) -> Self::Coordinates {
        (self.num_writes as u32, self.accumulator_length as u32)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos.0 < self.num_writes as u32 && pos.1 < self.accumulator_length as u32
    }
}

/// Build a [`ReduceOutputLayout`] from the output tensor and reduce/vec axes.
///
/// For simple reduces (`accumulator_length == 1`) `k_iter` never advances, so
/// the layout degenerates to `position = write_index` — a flat enumeration of
/// output vectors.
///
/// For topk-style reduces (`accumulator_length > 1`) the k slots live along
/// `reduce_axis` with stride `stride(reduce_axis) / vec` vectors, and the vec
/// axis (contiguous in the output, scalar stride 1) advances by one vector
/// per step, so `write_stride = 1`. When the two axes coincide — i.e. a
/// rank-1 output or any degenerate case where there is no separate SIMD axis
/// — `write_stride` collapses to `0` and `num_writes` to `1`, so every
/// `write_index` lands on the same k slot (should not matter because this collapse happens
/// only if we have only one unit).
#[cube]
pub(crate) fn build_reduce_output_layout<Out: NumericVector>(
    output: &VirtualTensor<Out::T, Out::N, ReadWrite>,
    reduce_axis: usize,
    out_vec_axis: usize,
    #[comptime] accumulator_length: usize,
) -> ReduceOutputLayout {
    let vec = output.vector_size();
    let num_vectored_reductions = output.shape(out_vec_axis) / vec;

    if comptime![accumulator_length == 1] {
        // Simple reduce: `write_index` is a flat vector offset into the
        // output, which matches the pre-topk behavior.
        ReduceOutputLayout::new(
            num_vectored_reductions,
            1,
            num_vectored_reductions,
            accumulator_length,
        )
    } else {
        let k_stride = output.stride(reduce_axis) / vec;
        // Branchless: `distinct` is 1 when reduce_axis != out_vec_axis, else 0.
        let distinct = usize::cast_from(reduce_axis != out_vec_axis);
        let write_stride = distinct;
        let num_writes = distinct * num_vectored_reductions + (1 - distinct);
        ReduceOutputLayout::new(k_stride, write_stride, num_writes, accumulator_length)
    }
}
