use std::marker::PhantomData;

use cubecl::cube;
use cubecl::prelude::*;

use crate::components::batch::{
    BatchConfig as _, BatchMatmul, BatchMatmulFamily, MatLayout,
    gemm_plane_parallel::{
        GemmPlaneParallelBlueprint, GemmPlaneParallelConfig, GemmPlaneParallelFamily,
        plane::execute_plane,
    },
};

use crate::{
    definition::{cube_pos_to_m_n_batch, *},
    launch::MatmulArgs,
};

#[cube(launch_unchecked, explicit_define, address_type = "dynamic")]
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
        Vector<Lhs, LhsSize>,
        Vector<Rhs, RhsSize>,
        Vector<Acc, AccSize>,
    >,
    output: &mut <Args as MatmulArgs>::Output<Vector<Acc, AccSize>>,
    runtime_config: (),
    cube_mapping: CubeMapping,
    #[comptime] blueprint: GemmPlaneParallelBlueprint,
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
    let config = comptime!(GemmPlaneParallelFamily::expand_config(
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

    GemmPlaneParallel::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs, LhsSize),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs, RhsSize),
        (Acc, AccSize, Acc, AccSize, RegisterAcc, AccSize),
    )>::execute::<Args>(&mut state, cube_mapping, config);
}

pub struct GemmPlaneParallel<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for GemmPlaneParallel<MP> {
    type Config = GemmPlaneParallelConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        let (_, m, k) = lhs.shape();
        let (_, _, n) = rhs.shape();

        // Cube grid is always laid out as `(m_cubes, n_cubes, batch)`. For
        // GEMV variants one of `m_cubes`/`n_cubes` is 1.
        let (cube_m, cube_n, batch_cube) = cube_pos_to_m_n_batch(&cube_mapping);

        let lhs_batch = Args::batch_lhs(state, batch_cube as usize);
        let rhs_batch = Args::batch_rhs(state, batch_cube as usize);
        let out_batch = Args::batch_out(state, batch_cube as usize);

        let vector_size = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let size!(N) = vector_size;

        let check_bounds = config.check_bounds;

        let lhs_view = lhs.view(MatLayout::new(lhs_batch, (m, k)));
        let rhs_view = rhs.view(MatLayout::new(rhs_batch, (k, n)));
        let out_view = out.view_mut(MatLayout::new(out_batch, (m, n)));

        // Planes within a cube split the N axis: each plane owns one
        // output column within the cube's M row.
        let m_id = cube_m;
        let n_id = cube_n * config.num_planes + UNIT_POS_Y;

        execute_plane::<LhsG<MP>, RhsG<MP>, AccG<MP>, AccRE<MP>, N>(
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
}
