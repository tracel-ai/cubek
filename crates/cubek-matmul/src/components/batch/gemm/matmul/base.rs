use std::marker::PhantomData;

use cubecl::cube;
use cubecl::prelude::*;

use crate::components::batch::gemm::mat_layout::MatLayout;
use crate::components::batch::gemm::matmul::dot::execute_dot;
use crate::components::batch::gemm::matmul::outer_product::execute_outer_product;
use crate::components::batch::{
    BatchConfig as _, BatchMatmul, BatchMatmulFamily,
    gemm::{GemmBlueprint, GemmConfig, GemmFamily, PlanesSplit, Variant},
};

use crate::{
    definition::{cube_pos_to_m_n_batch, *},
    launch::MatmulArgs,
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
    #[define(Lhs, Rhs, Acc)] _global: [StorageType; 3],
    #[define(LhsSize, RhsSize, AccSize)] _sizes: [usize; 3],
) {
    let mut state =
        Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
            inputs,
            output,
            runtime_config,
            blueprint.lhs_global_layout_config(),
            blueprint.rhs_global_layout_config(),
            blueprint.out_global_layout_config(),
        );

    let vector_size_lhs = Args::view_lhs(&state).vector_size();
    let vector_size_rhs = Args::view_rhs(&state).vector_size();
    let vector_size_out = Args::view_out(&mut state).vector_size();
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

    let mut state =
        Args::init_state::<Vector<Lhs, LhsSize>, Vector<Rhs, RhsSize>, Vector<Acc, AccSize>>(
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
    )>::execute::<Args>(&mut state, cube_mapping, config);
}

pub struct Gemm<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for Gemm<MP> {
    type Config = GemmConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let lhs = Args::view_lhs(&*state);
        let rhs = Args::view_rhs(&*state);
        let out = Args::view_out(state);

        let (_, m, k) = lhs.shape();
        let (_, _, n) = rhs.shape();

        let (cube_m, cube_n, batch_cube) = cube_pos_to_m_n_batch(&cube_mapping);

        let lhs_batch = Args::batch_lhs(&*state, batch_cube as usize);
        let rhs_batch = Args::batch_rhs(&*state, batch_cube as usize);
        let out_batch = Args::batch_out(&*state, batch_cube as usize);

        let vector_size = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let size!(N) = vector_size;

        let check_bounds = config.check_bounds;
        let variant = comptime!(config.kind.variant());

        let lhs_view = lhs.view(MatLayout::new(lhs_batch, (m, k)));
        let rhs_view = rhs.view(MatLayout::new(rhs_batch, (k, n)));
        let out_view = out.view_mut(MatLayout::new(out_batch, (m, n)));

        // Map cube + plane coords to the per-plane block origin. Per
        // variant, each axis is enumerated in *blocks* — the split axis
        // is `cube_axis * num_planes + plane_id`, the non-split axis is
        // just `cube_axis`. A block is 1 cell for the axis that the
        // variant covers per-plane (Dot covers 1×1; OuterN covers 1
        // M-row × NR N-cols; OuterM covers MR M-rows × 1 N-col), and
        // `vector_size` cells for the axis it walks in a block.
        let vs_u32 = comptime!(vector_size as u32);

        let (m_id, n_id) = match comptime!(config.planes_split) {
            PlanesSplit::M => {
                // OuterN splits M (and only OuterN uses split_M in v1).
                // Plane covers one M-row at `block_id`; cube_n indexes
                // NR-blocks along N → `n_pos_base = cube_n * vs`.
                let block_id = cube_m * config.num_planes + UNIT_POS_Y;
                match comptime!(variant) {
                    Variant::OuterNLhsContig | Variant::OuterNLhsStrided => {
                        (block_id, cube_n * vs_u32)
                    }
                    Variant::OuterM => (block_id * vs_u32, cube_n),
                    Variant::Dot => (block_id, cube_n),
                }
            }
            PlanesSplit::N => {
                // OuterM and Dot split N. OuterM: plane covers one
                // N-col at `block_id`; cube_m indexes MR-blocks along M.
                // Dot: 1×1 per plane.
                let block_id = cube_n * config.num_planes + UNIT_POS_Y;
                match comptime!(variant) {
                    Variant::OuterNLhsContig | Variant::OuterNLhsStrided => {
                        (cube_m, block_id * vs_u32)
                    }
                    Variant::OuterM => (cube_m * vs_u32, block_id),
                    Variant::Dot => (cube_m, block_id),
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
            Variant::OuterNLhsContig => {
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
                    true,
                    false,
                    check_bounds,
                );
            }
            Variant::OuterNLhsStrided => {
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
                    true,
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
                    false,
                    false,
                    check_bounds,
                );
            }
        }
    }
}
