use std::marker::PhantomData;

use cubecl::cube;
use cubecl::prelude::*;

use crate::{
    definition::*,
    multi_level::{
        args::MatmulArgs,
        components::batch::{
            BatchConfig as _, BatchMatmul, BatchMatmulFamily,
            gemm::{
                GemmBlueprint, GemmConfig, GemmFamily, PlanesSplit, Variant,
                mat_layout::MatLayout,
                matmul::{dot::execute_dot, outer_product::execute_outer_product},
            },
        },
        definition::{cube_pos_to_m_n_batch, *},
    },
};

#[cube(launch_unchecked, explicit_define, address_type = "dynamic")]
#[allow(clippy::type_complexity)]
/// Launches the matmul kernel
pub fn matmul_entry<
    Args: MatmulArgs<Config = ()>,
    Lhs: Numeric,
    LhsSize: Size,
    Rhs: Numeric,
    RhsSize: Size,
    Acc: Numeric,
    AccSize: Size,
>(
    inputs: &<Args as MatmulArgs>::Input<
        Vector<Lhs, LhsSize>,
        Vector<Rhs, RhsSize>,
        Vector<Acc, AccSize>,
    >,
    output: &mut <Args as MatmulArgs>::Output<Vector<Acc, AccSize>>,
    runtime_config: (),
    cube_mapping: CubeMapping,
    #[comptime] blueprint: GemmBlueprint,
    #[define(Lhs, Rhs, Acc)] _global: [ElemType; 3],
    #[define(LhsSize, RhsSize, AccSize)] _sizes: [usize; 3],
) {
    let state = Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
        inputs,
        output,
        runtime_config,
        blueprint.lhs_global_layout_config(),
        blueprint.rhs_global_layout_config(),
        blueprint.out_global_layout_config(),
    );

    let vector_size_lhs = Args::view_lhs(&state).vector_size();
    let vector_size_rhs = Args::view_rhs(&state).vector_size();
    let vector_size_out = Args::view_out(&state).vector_size();
    let vector_sizes = comptime!(MatmulVectorSizes {
        lhs: vector_size_lhs,
        rhs: vector_size_rhs,
        out: vector_size_out,
    });

    let device_props = comptime::device_properties();
    let config = comptime!(GemmFamily::expand_config(
        &device_props,
        &blueprint,
        &blueprint.dtypes,
        &vector_sizes
    ));

    if comptime!(config.is_err()) {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }
    let config = comptime!(config.unwrap());

    let state = Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
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

    Gemm::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs, LhsSize),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs, RhsSize),
        (Acc, AccSize, Acc, AccSize, RegisterAcc, AccSize),
    )>::execute::<Args>(&state, cube_mapping, config);
}

pub struct Gemm<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for Gemm<MP> {
    type Config = GemmConfig;

    fn execute<Args: MatmulArgs>(
        state: &Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        let (_, m, k) = lhs.shape();
        let (_, _, n) = rhs.shape();

        let (cube_m, cube_n, batch_cube) = cube_pos_to_m_n_batch(&cube_mapping);

        let lhs_batch = Args::batch_lhs(state, batch_cube as usize);
        let rhs_batch = Args::batch_rhs(state, batch_cube as usize);
        let out_batch = Args::batch_out(state, batch_cube as usize);

        let lhs_vec = lhs.vector_size();
        let rhs_vec = rhs.vector_size();
        let vector_size = comptime![Ord::max(lhs_vec, rhs_vec)];
        let size!(N) = vector_size;

        let check_bounds = config.check_bounds;
        let variant = comptime!(config.variant);

        let lhs_view = lhs.view(MatLayout::new(lhs_batch, (m, k)));
        let rhs_view = rhs.view(MatLayout::new(rhs_batch, (k, n)));
        let out_view = out.view_mut(MatLayout::new(out_batch, (m, n)));

        // Map cube + plane coords to the per-plane block origin. Per
        // variant, each axis is enumerated in *blocks*: the split axis
        // is `cube_axis * num_planes + plane_id`, the non-split axis is
        // just `cube_axis`. A block is `block_cells` along the axis its
        // accumulators run (N for OuterN, M for OuterM, the split axis
        // for Dot) and 1 cell along the other.
        let accumulators = config.accumulators;
        let block_cells =
            comptime!(variant.cells_per_accumulator(vector_size) as u32 * accumulators);

        let (m_id, n_id) = match comptime!(config.planes_split) {
            PlanesSplit::M => {
                let block_id = cube_m * config.num_planes + UNIT_POS_Y;
                match comptime!(variant) {
                    Variant::OuterN => (block_id, cube_n * block_cells),
                    Variant::OuterM | Variant::Dot => (block_id * block_cells, cube_n),
                }
            }
            PlanesSplit::N => {
                let block_id = cube_n * config.num_planes + UNIT_POS_Y;
                match comptime!(variant) {
                    Variant::OuterM => (cube_m * block_cells, block_id),
                    Variant::OuterN | Variant::Dot => (cube_m, block_id * block_cells),
                }
            }
        };

        match comptime!(variant) {
            Variant::Dot => {
                execute_dot::<LhsG<MP>, RhsG<MP>, AccG<MP>, AccRE<MP>, N>(
                    lhs_view,
                    rhs_view,
                    out_view,
                    m_id,
                    n_id,
                    k,
                    config.plane_dim,
                    vector_size as u32,
                    check_bounds,
                );
            }
            Variant::OuterN => {
                execute_outer_product::<
                    Global<Lhs<MP>>,
                    Global<Rhs<MP>>,
                    AccG<MP>,
                    AccRE<MP>,
                    GlobalSize<Lhs<MP>>,
                    GlobalSize<Rhs<MP>>,
                    N,
                >(
                    lhs_view,
                    rhs_view,
                    out_view,
                    m_id,
                    n_id,
                    k,
                    vector_size as u32,
                    accumulators,
                    true,
                    check_bounds,
                );
            }
            Variant::OuterM => {
                execute_outer_product::<
                    Global<Lhs<MP>>,
                    Global<Rhs<MP>>,
                    AccG<MP>,
                    AccRE<MP>,
                    GlobalSize<Lhs<MP>>,
                    GlobalSize<Rhs<MP>>,
                    N,
                >(
                    lhs_view,
                    rhs_view,
                    out_view,
                    m_id,
                    n_id,
                    k,
                    vector_size as u32,
                    accumulators,
                    false,
                    check_bounds,
                );
            }
        }
    }
}
