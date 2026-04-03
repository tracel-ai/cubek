use std::marker::PhantomData;

use crate::components::batch::{
    BatchConfig as _, BatchMatmul, BatchMatmulFamily,
    gemv_plane_parallel::{
        GemvPlan, GemvPlaneParallelBlueprint, VecMatPlaneParallelConfig,
        GemvPlaneParallelFamily,
        layout::{MatLayout, VecLayout},
    },
};

use crate::{definition::*, launch::MatmulArgs};
use cubecl::{
    cube,
    num_traits::Zero,
    std::tensor::layout::{Coords1d, Coords2d},
};
use cubecl::{prelude::*, std::tensor::View};
use cubek_std::MatrixLayout;

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
    #[comptime] blueprint: GemvPlaneParallelBlueprint,
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
    let config = comptime!(GemvPlaneParallelFamily::expand_config(
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

    VecMatPlaneParallel::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs),
        (Acc, AccSize, Acc, AccSize, RegisterAcc),
    )>::execute::<Args>(&mut state, cube_mapping, config);
}

pub struct VecMatPlaneParallel<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for VecMatPlaneParallel<MP> {
    type Config = VecMatPlaneParallelConfig;

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
        let (_, matrix_cube, batch_cube) = cube_mapping.cube_pos_to_tensor_pos();

        let lhs_batch = Args::batch_lhs(state, batch_cube as usize);
        let rhs_batch = Args::batch_rhs(state, batch_cube as usize);
        let out_batch = Args::batch_out(state, batch_cube as usize);

        match config.plan {
            GemvPlan::VecMatDirect => execute_gemv::<LhsG<MP>, RhsG<MP>, AccG<MP>, AccR<MP>>(
                lhs.view(VecLayout::new(lhs_batch, k as usize)),
                rhs.view(MatLayout::new(rhs_batch, (k, n), MatrixLayout::ColMajor)),
                out.view_mut(VecLayout::new(out_batch, n as usize)),
                matrix_cube,
                k,
                config.num_planes,
                config.plane_dim,
            ),
            GemvPlan::VecMatTransposeSwap => {
                execute_gemv::<LhsG<MP>, RhsG<MP>, AccG<MP>, AccR<MP>>(
                    lhs.view(VecLayout::new(lhs_batch, k as usize)),
                    rhs.view(MatLayout::new(rhs_batch, (k, n), MatrixLayout::RowMajor)),
                    out.view_mut(VecLayout::new(out_batch, n as usize)),
                    matrix_cube,
                    k,
                    config.num_planes,
                    config.plane_dim,
                )
            }
            GemvPlan::MatVecDirect => execute_gemv::<RhsG<MP>, LhsG<MP>, AccG<MP>, AccR<MP>>(
                rhs.view(VecLayout::new(rhs_batch, k as usize)),
                lhs.view(MatLayout::new(lhs_batch, (m, k), MatrixLayout::RowMajor)),
                out.view_mut(VecLayout::new(out_batch, m as usize)),
                matrix_cube,
                k,
                config.num_planes,
                config.plane_dim,
            ),
            GemvPlan::MatVecTransposeSwap => {
                execute_gemv::<RhsG<MP>, LhsG<MP>, AccG<MP>, AccR<MP>>(
                    rhs.view(VecLayout::new(rhs_batch, k as usize)),
                    lhs.view(MatLayout::new(lhs_batch, (m, k), MatrixLayout::ColMajor)),
                    out.view_mut(VecLayout::new(out_batch, m as usize)),
                    matrix_cube,
                    k,
                    config.num_planes,
                    config.plane_dim,
                )
            }
        }
    }
}

#[cube]
fn execute_gemv<V: CubePrimitive, M: CubePrimitive, O: CubePrimitive, AccR: Numeric>(
    vec: View<V, Coords1d>,
    mat: View<M, Coords2d>,
    out: View<O, Coords1d, ReadWrite>,
    cube_id: u32,
    k_dim: u32,
    #[comptime] num_planes: u32,
    #[comptime] plane_dim: u32,
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;

    let mn_pos = cube_id * num_planes + plane_id;

    let size!(N) = comptime![Ord::max(vec.vector_size(), mat.vector_size())];
    let vector_size = N::value() as u32;
    let tile_size = plane_dim * vector_size;
    let num_tiles = k_dim / tile_size;

    let mut acc = Vector::<AccR, N>::zero();

    for tile_index in 0..num_tiles {
        let swizzled_tile_index = (tile_index + plane_id) % num_tiles;
        let k_base = swizzled_tile_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;
        let vec_val = vec.read_checked(k_pos as usize);
        let mat_val = mat.read_checked((mn_pos, k_pos));

        acc += Vector::cast_from(vec_val) * Vector::cast_from(mat_val);
    }

    let mut sum = AccR::zero();

    #[unroll]
    for i in 0..N::value() {
        sum += acc[i];
    }

    let sum = if comptime!(plane_dim > 1) {
        plane_sum(sum)
    } else {
        sum
    };

    if unit_id == 0 {
        out.write_checked(mn_pos as usize, O::cast_from(sum));
    }
}
