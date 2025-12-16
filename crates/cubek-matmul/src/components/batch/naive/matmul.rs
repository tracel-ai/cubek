use std::marker::PhantomData;

use crate::definition::MatrixLayout;
use crate::{
    components::{
        batch::{BatchConfig, BatchMatmul, naive::NaiveMatmulConfig},
        global::{GlobalReaderConfig, GlobalWriterConfig},
    },
    definition::*,
    launch::MatmulArgs,
};
use cubecl::prelude::*;
use cubecl::std::tensor::{View, layout::Coords3d};
use cubecl::{CubeDim, cube};

// #[cube(launch_unchecked)]
// fn naive_matmul_entry<I: Numeric, M: Numeric, O: Numeric>(
//     lhs: &View<Line<I>, Coords3d>,
//     rhs: &View<Line<I>, Coords3d>,
//     out: &mut Tensor<O>,
//     #[define(I)] _input_dtype: StorageType,
//     #[define(M)] _acc_dtype: StorageType,
//     #[define(O)] _output_dtype: StorageType,
// ) {
//     let rank = out.rank();

//     let (_, _, k) = lhs.shape();
//     let size_m = out.shape(rank - 2);
//     let size_n = out.shape(rank - 1);

//     let batch = ABSOLUTE_POS_Z;
//     let m = ABSOLUTE_POS_X;
//     let n = ABSOLUTE_POS_Y;

//     if m >= size_m || n >= size_n {
//         terminate!();
//     }

//     let offset_out = batch * out.stride(rank - 2) * out.shape(rank - 2);

//     let line_size = comptime![Ord::max(lhs.line_size(), rhs.line_size())];
//     let mut sum = Line::empty(line_size).fill(O::from_int(0));

//     for k in range_stepped(0, k, line_size) {
//         let lhs = load_unrolled(lhs, (batch, m, k), MatrixLayout::RowMajor, line_size);
//         let rhs = load_unrolled(rhs, (batch, k, n), MatrixLayout::ColMajor, line_size);

//         sum += Line::cast_from(Line::<M>::cast_from(lhs) * Line::<M>::cast_from(rhs));
//     }

//     let mut out_index = m * out.stride(rank - 2) + n;
//     out_index += offset_out;

//     let unroll_sum = line_size != 1;
//     if unroll_sum {
//         let mut accum = O::from_int(0);
//         // we unroll the loop to sum `vectorization_factor` elements at once, which lets us
//         // use SIMD instructions to speed up the computation
//         #[unroll]
//         for v in 0..line_size {
//             accum += sum[v];
//         }

//         out[out_index] = accum;
//     } else {
//         out[out_index] = sum[0];
//     }
// }

#[cube(launch_unchecked)]
/// Launches the matmul kernel
pub(crate) fn matmul_entry<
    Args: MatmulArgs,
    LhsG: Numeric,
    RhsG: Numeric,
    AccG: Numeric,
    LhsS: Numeric,
    RhsS: Numeric,
    AccS: Numeric,
    LhsR: Numeric,
    RhsR: Numeric,
    AccR: Numeric,
>(
    inputs: &<Args as MatmulArgs>::Input<LhsG, RhsG, AccG>,
    output: &mut <Args as MatmulArgs>::Output<AccG>,
    cube_count_args: CubeCountInput,
    #[comptime] config: NaiveMatmulConfig,
    #[define(LhsG, RhsG, AccG)] _global: [StorageType; 3],
    #[define(LhsS, RhsS, AccS)] _stage: [StorageType; 3],
    #[define(LhsR, RhsR, AccR)] _register: [StorageType; 3],
) {
    #[allow(clippy::collapsible_if)]
    if comptime!(config.can_yield_extra_cubes()) {
        if CUBE_POS >= cube_count_args.num_valid_cubes() {
            terminate!()
        }
    }

    let mut state = Args::init_state::<LhsG, RhsG, AccG>(inputs, output);

    NaiveMatmul::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::execute::<Args>(
        &mut state,
        cube_count_args,
        config,
    );
}

pub struct NaiveMatmul<MP: MatmulPrecision> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulPrecision> BatchMatmul<MP> for NaiveMatmul<MP> {
    type Config = NaiveMatmulConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        _cube_count_args: CubeCountInput,
        #[comptime] _config: Self::Config,
    ) {
        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let mut out = Args::view_out(state);

        let (_, _, k) = lhs.shape();
        let (_, size_m, size_n) = out.shape();

        let batch = ABSOLUTE_POS_Z;
        let m = ABSOLUTE_POS_X;
        let n = ABSOLUTE_POS_Y;

        if m >= size_m || n >= size_n {
            terminate!();
        }

        let line_size = comptime![Ord::max(lhs.line_size(), rhs.line_size())];
        let mut sum = Line::empty(line_size).fill(<AccG<MP> as Numeric>::from_int(0));

        for k in range_stepped(0u32, k, line_size) {
            let lhs = load_unrolled(&lhs, (batch, m, k), MatrixLayout::RowMajor, line_size);
            let rhs = load_unrolled(&rhs, (batch, k, n), MatrixLayout::ColMajor, line_size);

            sum += Line::cast_from(
                Line::<AccR<MP>>::cast_from(lhs) * Line::<AccR<MP>>::cast_from(rhs),
            );
        }

        let unroll_sum = line_size != 1u32;
        if unroll_sum {
            let mut accum = <AccG<MP> as Numeric>::from_int(0);
            // we unroll the loop to sum `vectorization_factor` elements at once, which lets us
            // use SIMD instructions to speed up the computation
            #[unroll]
            for v in 0u32..line_size {
                accum += sum[v];
            }

            out[(batch, m, n)] = Line::empty(1u32).fill(accum);
        } else {
            out[(batch, m, n)] = Line::empty(1u32).fill(sum[0u32]);
        }
    }
}

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
