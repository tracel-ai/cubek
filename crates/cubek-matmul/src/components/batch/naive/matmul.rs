use std::marker::PhantomData;

use crate::components::batch::{BatchConfig as _, SliceIndex};
use crate::definition::MatrixLayout;
use crate::{
    components::batch::{BatchMatmul, naive::NaiveMatmulConfig},
    definition::*,
    launch::MatmulArgs,
};
use cubecl::cube;
use cubecl::prelude::*;
use cubecl::std::tensor::View;
use cubecl::std::tensor::layout::Coords2d;

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
    let mut state = Args::init_state::<LhsG, RhsG, AccG>(
        inputs,
        output,
        config.lhs_global_layout_config(),
        config.rhs_global_layout_config(),
        config.out_global_layout_config(),
    );

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
        let out = Args::view_out(state);

        let (_, _, k) = lhs.shape();
        let (_, size_m, size_n) = out.shape();

        let m = ABSOLUTE_POS_X;
        let n = ABSOLUTE_POS_Y;
        let batch = ABSOLUTE_POS_Z;

        let lhs_batch = Args::batch_lhs(state, batch);
        let lhs = lhs.view(SliceIndex::new(lhs_batch, lhs.shape()));
        let rhs_batch = Args::batch_rhs(state, batch);
        let rhs = rhs.view(SliceIndex::new(rhs_batch, rhs.shape()));
        let out_batch = Args::batch_out(state, batch);
        let mut out = out.view_mut(SliceIndex::new(out_batch, out.shape()));

        if m >= size_m || n >= size_n {
            terminate!();
        }

        let line_size = comptime![Ord::max(lhs.line_size(), rhs.line_size())];
        let mut sum = Line::empty(line_size).fill(<AccG<MP> as Numeric>::from_int(0));

        for k in range_stepped(0u32, k, line_size) {
            let lhs = load_unrolled(&lhs, (m, k), MatrixLayout::RowMajor, line_size);
            let rhs = load_unrolled(&rhs, (k, n), MatrixLayout::ColMajor, line_size);

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

            out[(m, n)] = Line::empty(1u32).fill(accum);
            // out[(m, n)] = Line::cast_from(tmp);
        } else {
            out[(m, n)] = Line::empty(1u32).fill(sum[0u32]);
        }
    }
}

#[cube]
fn load_unrolled<I: Numeric>(
    view: &View<Line<I>, Coords2d>,
    pos: Coords2d,
    #[comptime] layout: MatrixLayout,
    #[comptime] line_size: u32,
) -> Line<I> {
    comptime![assert!(line_size >= view.line_size())];
    let view_line_size = view.line_size();
    if comptime![view.line_size() == line_size] {
        view[pos]
    } else {
        let (row, col) = pos;
        let mut out = Line::empty(line_size);
        #[unroll]
        for i in range_stepped(0, line_size, view_line_size) {
            let pos = match layout {
                MatrixLayout::RowMajor => (row, col + i),
                MatrixLayout::ColMajor => (row + i, col),
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
