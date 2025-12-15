use cubecl::prelude::*;

use crate::definition::MatrixLayout;
use cubecl::std::tensor::{View, layout::Coords3d};

#[cube]
fn load_unrolled<I: Numeric>(
    view: &View<Line<I>, Coords3d>,
    pos: Coords3d,
    #[comptime] layout: MatrixLayout,
    #[comptime] line_size: u32,
) -> Line<I> {
    comptime![assert!(line_size >= view.line_size())];
    let view_line_size = view.line_size();
    if comptime![view.line_size() == line_size] {
        view[pos]
    } else {
        let (b, row, col) = pos;
        let mut out = Line::empty(line_size);
        #[unroll]
        for i in range_stepped(0, line_size, view_line_size) {
            let pos = match layout {
                MatrixLayout::RowMajor => (b, row, col + i),
                MatrixLayout::ColMajor => (b, row + i, col),
            };
            let value = view[pos];
            #[unroll]
            for n in 0..view_line_size {
                out[i + n] = value[n];
            }
        }
        out
    }
}

#[cube(launch_unchecked)]
pub fn naive_matmul<I: Numeric, M: Numeric, O: Numeric>(
    lhs: &View<Line<I>, Coords3d>,
    rhs: &View<Line<I>, Coords3d>,
    out: &mut Tensor<O>,
    #[define(I)] _input_dtype: StorageType,
    #[define(M)] _acc_dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
) {
    let rank = out.rank();

    let (_, _, k) = lhs.shape();
    let size_m = out.shape(rank - 2);
    let size_n = out.shape(rank - 1);

    let batch = ABSOLUTE_POS_Z;
    let m = ABSOLUTE_POS_X;
    let n = ABSOLUTE_POS_Y;

    if m >= size_m || n >= size_n {
        terminate!();
    }

    let offset_out = batch * out.stride(rank - 2) * out.shape(rank - 2);

    let line_size = comptime![Ord::max(lhs.line_size(), rhs.line_size())];
    let mut sum = Line::empty(line_size).fill(O::from_int(0));

    for k in range_stepped(0, k, line_size) {
        let lhs = load_unrolled(lhs, (batch, m, k), MatrixLayout::RowMajor, line_size);
        let rhs = load_unrolled(rhs, (batch, k, n), MatrixLayout::ColMajor, line_size);

        sum += Line::cast_from(Line::<M>::cast_from(lhs) * Line::<M>::cast_from(rhs));
    }

    let mut out_index = m * out.stride(rank - 2) + n;
    out_index += offset_out;

    let unroll_sum = line_size != 1;
    if unroll_sum {
        let mut accum = O::from_int(0);
        // we unroll the loop to sum `vectorization_factor` elements at once, which lets us
        // use SIMD instructions to speed up the computation
        #[unroll]
        for v in 0..line_size {
            accum += sum[v];
        }

        out[out_index] = accum;
    } else {
        out[out_index] = sum[0];
    }
}
