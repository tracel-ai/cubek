use std::marker::PhantomData;

use crate::components::batch::base::BatchMatmulFamily;
use crate::components::batch::naive::{NaiveBatchMatmulFamily, NaiveBlueprint};
use crate::components::batch::{BatchConfig as _, SliceIndex};

use crate::{
    components::batch::{BatchMatmul, naive::NaiveMatmulConfig},
    definition::*,
    launch::MatmulArgs,
};
use cubecl::prelude::*;
use cubecl::std::tensor::View;
use cubecl::std::tensor::layout::Coords2d;
use cubecl::{cube, num_traits::Zero};
use cubek_std::MatrixLayout;

#[cube(launch_unchecked, address_type = "dynamic")]
#[allow(clippy::type_complexity)]
/// Launches the matmul kernel
pub(crate) fn matmul_entry<
    Args: MatmulArgs<Config = ()>,
    Lhs: Numeric,
    LhsSize: Size,
    Rhs: Numeric,
    RhsSize: Size,
    Acc: Numeric,
    AccSize: Size,
>(
    inputs: &<Args as MatmulArgs>::Input<
        Line<Lhs, LhsSize>,
        Line<Rhs, RhsSize>,
        Line<Acc, AccSize>,
    >,
    output: &mut <Args as MatmulArgs>::Output<Line<Acc, AccSize>>,
    runtime_config: (),
    cube_mapping: CubeMapping,
    #[comptime] blueprint: NaiveBlueprint,
    #[define(Lhs, Rhs, Acc)] _global: [StorageType; 3],
    #[define(LhsSize, RhsSize, AccSize)] _sizes: [usize; 3],
) {
    let mut state = Args::init_state::<Line<Lhs, LhsSize>, Line<Rhs, RhsSize>, Line<Acc, AccSize>>(
        inputs,
        output,
        runtime_config,
        blueprint.lhs_global_layout_config(),
        blueprint.rhs_global_layout_config(),
        blueprint.out_global_layout_config(),
    );

    let line_size_lhs = Args::view_lhs(&state).line_size();
    let line_size_rhs = Args::view_rhs(&state).line_size();
    let line_size_out = Args::view_out(&mut state).line_size();
    let line_sizes = comptime!(MatmulLineSizes {
        lhs: line_size_lhs,
        rhs: line_size_rhs,
        out: line_size_out,
    });

    let device_props = comptime::device_properties();
    let config = comptime!(NaiveBatchMatmulFamily::expand_config(
        &device_props,
        &blueprint,
        &blueprint.dtypes,
        &line_sizes
    ));

    if comptime!(config.is_err()) {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }
    let config = comptime!(config.unwrap());

    let mut state = Args::init_state::<Line<Lhs, LhsSize>, Line<Rhs, RhsSize>, Line<Acc, AccSize>>(
        inputs,
        output,
        runtime_config,
        config.lhs_global_layout_config(),
        config.rhs_global_layout_config(),
        config.out_global_layout_config(),
    );

    let define!(RegisterLhs) = blueprint.dtypes.lhs_register;
    let define!(RegisterRhs) = blueprint.dtypes.rhs_register;
    let define!(RegisterAcc) = blueprint.dtypes.acc_register;

    NaiveMatmul::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs),
        (Acc, AccSize, Acc, AccSize, RegisterAcc),
    )>::execute::<Args>(&mut state, cube_mapping, config);
}

pub struct NaiveMatmul<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for NaiveMatmul<MP> {
    type Config = NaiveMatmulConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        _cube_mapping: CubeMapping,
        #[comptime] _config: Self::Config,
    ) {
        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        let (_, _, k) = lhs.shape();
        let (_, size_m, size_n) = out.shape();

        let m = ABSOLUTE_POS_X;
        let n = ABSOLUTE_POS_Y;
        let batch = ABSOLUTE_POS_Z as usize;

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
        let size!(NA) = line_size;
        let mut sum = Line::<AccR<MP>, NA>::zero();

        for k in range_stepped(0u32, k, line_size as u32) {
            let lhs = load_unrolled::<_, _, NA>(&lhs, (m, k), MatrixLayout::RowMajor);
            let rhs = load_unrolled::<_, _, NA>(&rhs, (k, n), MatrixLayout::ColMajor);

            sum += Line::cast_from(
                Line::<AccR<MP>, NA>::cast_from(lhs) * Line::<AccR<MP>, NA>::cast_from(rhs),
            );
        }

        let unroll_sum = line_size != 1usize;
        if unroll_sum {
            let mut accum = AccR::<MP>::zero();
            // we unroll the loop to sum `vectorization_factor` elements at once, which lets us
            // use SIMD instructions to speed up the computation
            #[unroll]
            for v in 0..line_size {
                accum += sum[v];
            }

            out[(m, n)] = Line::cast_from(accum);
        } else {
            out[(m, n)] = Line::cast_from(sum[0]);
        }
    }
}

#[cube]
fn load_unrolled<I: Numeric, N: Size, N2: Size>(
    view: &View<Line<I, N>, Coords2d>,
    pos: Coords2d,
    #[comptime] layout: MatrixLayout,
) -> Line<I, N2> {
    let line_size = N2::value();
    comptime![assert!(line_size >= view.line_size())];
    let view_line_size = view.line_size();
    if view.line_size().comptime() == line_size {
        Line::cast_from(view[pos])
    } else {
        let (row, col) = pos;
        let mut out = Line::empty();
        #[unroll]
        for i in range_stepped(0, line_size as u32, view_line_size as u32) {
            let pos = match layout {
                MatrixLayout::RowMajor => (row, col + i),
                MatrixLayout::ColMajor => (row + i, col),
            };
            let value = view[pos];
            #[unroll]
            for n in 0..view_line_size {
                out[i as usize + n] = value[n];
            }
        }
        out
    }
}
