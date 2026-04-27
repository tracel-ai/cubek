use std::marker::PhantomData;

use crate::components::batch::{
    BatchConfig as _, BatchMatmul, BatchMatmulFamily, CheckBounds,
    gemv_plane_parallel::{
        GemvKind, GemvPlaneParallelBlueprint, GemvPlaneParallelFamily, VecMatPlaneParallelConfig,
        layout::{MatLayout, VecLayout},
    },
};

use crate::{
    definition::{cube_pos_to_matrix_batch, *},
    launch::MatmulArgs,
};
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
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs, LhsSize),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs, RhsSize),
        (Acc, AccSize, Acc, AccSize, RegisterAcc, AccSize),
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
        let (matrix_cube, batch_cube) = cube_pos_to_matrix_batch(&cube_mapping);

        let lhs_batch = Args::batch_lhs(state, batch_cube as usize);
        let rhs_batch = Args::batch_rhs(state, batch_cube as usize);
        let out_batch = Args::batch_out(state, batch_cube as usize);

        let vector_size = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let size!(N) = vector_size;

        let check_bounds = config.check_bounds;

        match config.plan {
            GemvKind::VecMatColMajor => execute_gemv::<LhsG<MP>, RhsG<MP>, AccG<MP>, AccR<MP>, N>(
                lhs.view(VecLayout::new(lhs_batch, k as usize)),
                rhs.view(MatLayout::new(rhs_batch, (k, n))),
                out.view_mut(VecLayout::new(out_batch, n as usize)),
                matrix_cube,
                k,
                config.num_planes,
                config.plane_dim,
                vector_size as u32,
                MatrixLayout::ColMajor,
                check_bounds,
            ),
            GemvKind::VecMatRowMajor => execute_gemv_transposed::<
                Global<Lhs<MP>>,
                Global<Rhs<MP>>,
                AccG<MP>,
                AccR<MP>,
                Stage<Rhs<MP>>,
                GlobalSize<Lhs<MP>>,
                GlobalSize<Rhs<MP>>,
            >(
                lhs.view(VecLayout::new(lhs_batch, k as usize)),
                rhs.view(MatLayout::new(rhs_batch, (k, n))),
                out.view_mut(VecLayout::new(out_batch, n as usize)),
                matrix_cube * config.num_planes + UNIT_POS_Y,
                k,
                vector_size as u32,
                MatrixLayout::RowMajor,
                check_bounds,
            ),
            GemvKind::MatVecRowMajor => execute_gemv::<RhsG<MP>, LhsG<MP>, AccG<MP>, AccR<MP>, N>(
                rhs.view(VecLayout::new(rhs_batch, k as usize)),
                lhs.view(MatLayout::new(lhs_batch, (m, k))),
                out.view_mut(VecLayout::new(out_batch, m as usize)),
                matrix_cube,
                k,
                config.num_planes,
                config.plane_dim,
                vector_size as u32,
                MatrixLayout::RowMajor,
                check_bounds,
            ),
            GemvKind::MatVecColMajor => execute_gemv_transposed::<
                Global<Rhs<MP>>,
                Global<Lhs<MP>>,
                AccG<MP>,
                AccR<MP>,
                Stage<Lhs<MP>>,
                GlobalSize<Rhs<MP>>,
                GlobalSize<Lhs<MP>>,
            >(
                rhs.view(VecLayout::new(rhs_batch, k as usize)),
                lhs.view(MatLayout::new(lhs_batch, (m, k))),
                out.view_mut(VecLayout::new(out_batch, m as usize)),
                matrix_cube * config.num_planes + UNIT_POS_Y,
                k,
                vector_size as u32,
                MatrixLayout::ColMajor,
                check_bounds,
            ),
        }
    }
}

#[cube]
fn execute_gemv<V: CubePrimitive, M: CubePrimitive, O: CubePrimitive, AccR: Numeric, N: Size>(
    vec: View<V, Coords1d>,
    mat: View<M, Coords2d>,
    out: View<O, Coords1d, ReadWrite>,
    cube_id: u32,
    k_dim: u32,
    #[comptime] num_planes: u32,
    #[comptime] plane_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] check_bounds: CheckBounds,
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;

    let mn_pos = cube_id * num_planes + plane_id;

    // The first if statement is running at comptime.
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        // This is a runtime cond.
        let should_terminate = mn_pos as usize >= out.shape();
        if should_terminate {
            terminate!();
        }
    }

    let segment_size = plane_dim * vector_size;
    let num_segments_k = k_dim / segment_size;

    let mut acc = Vector::<AccR, N>::zero();

    for segment_index in 0..num_segments_k {
        let swizzled_segment_index = (segment_index + plane_id) % num_segments_k;
        let k_base = swizzled_segment_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;

        let vec_val = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            vec.read_checked(k_pos as usize)
        } else {
            vec.read_unchecked(k_pos as usize)
        };

        let mat_val = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            match matrix_layout {
                MatrixLayout::RowMajor => mat.read_checked((mn_pos, k_pos)),
                MatrixLayout::ColMajor => mat.read_checked((k_pos, mn_pos)),
            }
        } else {
            match matrix_layout {
                MatrixLayout::RowMajor => mat.read_unchecked((mn_pos, k_pos)),
                MatrixLayout::ColMajor => mat.read_unchecked((k_pos, mn_pos)),
            }
        };

        acc += Vector::cast_from(vec_val) * Vector::cast_from(mat_val);
    }

    let sum = Vector::vector_sum(acc);

    let sum = if comptime!(plane_dim > 1) {
        O::cast_from(plane_sum(sum))
    } else {
        O::cast_from(sum)
    };

    let write_index = mn_pos as usize;

    #[allow(clippy::collapsible_else_if)]
    if comptime!(plane_dim == 1) {
        if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            out.write_checked(write_index, sum);
        } else {
            out.write(write_index, sum);
        }
    } else {
        if unit_id == 0 {
            if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
                out.write_checked(write_index, sum);
            } else {
                out.write(write_index, sum);
            }
        }
    }
}

#[cube]
fn execute_gemv_transposed<
    V: Scalar,
    M: Scalar,
    O: CubePrimitive,
    AccR: Numeric,
    SM: Scalar,
    VS: Size,
    MS: Size,
>(
    vec: View<Vector<V, VS>, Coords1d>,
    mat: View<Vector<M, MS>, Coords2d>,
    out: View<O, Coords1d, ReadWrite>,
    mn_id: u32,
    k_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] check_bounds: CheckBounds,
) {
    let mn_pos = mn_id * vector_size;

    // The first if statement is running at comptime.
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        // This is a runtime cond.
        let should_terminate = mn_pos as usize >= out.shape();
        if should_terminate {
            terminate!();
        }
    }

    let num_tiles_k = k_dim / vector_size;

    let mut accs: Array<Vector<AccR, VS>> = Array::new(vector_size as usize);
    for segment_iter in 0..vector_size {
        accs[segment_iter as usize] = Vector::zero();
    }

    // VS x VS tile
    let mut tile = Array::<SM>::new((vector_size * vector_size) as usize);

    for tile_index in 0..num_tiles_k {
        let global_k_pos = tile_index * vector_size;

        let vec_val = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            vec.read_checked(global_k_pos as usize)
        } else {
            vec.read_unchecked(global_k_pos as usize)
        };

        for segment_iter in 0..vector_size {
            let vector = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
                match matrix_layout {
                    MatrixLayout::RowMajor => {
                        mat.read_checked((global_k_pos + segment_iter, mn_pos))
                    }
                    MatrixLayout::ColMajor => {
                        mat.read_checked((mn_pos, global_k_pos + segment_iter))
                    }
                }
            } else {
                match matrix_layout {
                    MatrixLayout::RowMajor => {
                        mat.read_unchecked((global_k_pos + segment_iter, mn_pos))
                    }
                    MatrixLayout::ColMajor => {
                        mat.read_unchecked((mn_pos, global_k_pos + segment_iter))
                    }
                }
            };

            #[unroll]
            for i in 0..vector_size {
                tile[(segment_iter * vector_size + i) as usize] = SM::cast_from(vector[i as usize]);
            }
        }

        for segment_iter in 0..vector_size {
            let mut mat_val: Vector<SM, VS> = Vector::empty();

            #[unroll]
            for i in 0..vector_size {
                mat_val[i as usize] = tile[(i * vector_size + segment_iter) as usize];
            }

            accs[segment_iter as usize] += Vector::cast_from(vec_val) * Vector::cast_from(mat_val);
        }
    }

    // Write back
    for segment_iter in 0..vector_size {
        let acc = accs[segment_iter as usize];
        let sum = O::cast_from(Vector::vector_sum(acc));
        let index = (mn_pos + segment_iter) as usize;

        if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            out.write_checked(index, sum);
        } else {
            out.write(index, sum);
        }
    }
}
