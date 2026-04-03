use std::marker::PhantomData;

use crate::components::batch::{
    BatchConfig as _, BatchMatmul, BatchMatmulFamily,
    gemv_plane_parallel::{
        GemvKind, GemvPlaneParallelBlueprint, GemvPlaneParallelFamily, VecMatPlaneParallelConfig,
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

        let vector_size = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let size!(N) = vector_size;

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
                matrix_cube,
                k,
                config.num_planes,
                config.plane_dim,
                vector_size as u32,
                MatrixLayout::RowMajor,
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
                matrix_cube,
                k,
                config.num_planes,
                config.plane_dim,
                vector_size as u32,
                MatrixLayout::ColMajor,
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
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;

    let mn_pos = cube_id * num_planes + plane_id;

    let segment_size = plane_dim * vector_size;
    let num_segments_k = k_dim / segment_size;

    let mut acc = Vector::<AccR, N>::zero();

    for segment_index in 0..num_segments_k {
        let swizzled_segment_index = (segment_index + plane_id) % num_segments_k;
        let k_base = swizzled_segment_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;
        let vec_val = vec.read_checked(k_pos as usize);

        let mat_val = match matrix_layout {
            // mat=lhs
            MatrixLayout::RowMajor => mat.read_checked((mn_pos, k_pos)),
            // mat=rhs
            MatrixLayout::ColMajor => mat.read_checked((k_pos, mn_pos)),
        };

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
    cube_id: u32,
    k_dim: u32,
    #[comptime] num_planes: u32,
    #[comptime] plane_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] matrix_layout: MatrixLayout,
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;

    let segment_size = comptime!(plane_dim * vector_size);
    let cube_offset = cube_id * segment_size;
    let num_segments_k = k_dim / segment_size;

    let segments_per_plane = segment_size / num_planes;

    let mut accs: Array<Vector<AccR, VS>> = Array::new(segments_per_plane as usize);
    for segment_iter in 0..segments_per_plane {
        accs[segment_iter as usize] = Vector::zero();
    }

    let mut smem = SharedMemory::<SM>::new((segment_size * segment_size) as usize);

    for segment_index in 0..num_segments_k {
        let k_base = segment_index * plane_dim;

        let local_k_pos = unit_id * vector_size;
        let global_k_pos = k_base * vector_size + local_k_pos;
        let vec_val = vec.read_checked(global_k_pos as usize);

        assert!(segment_size.is_multiple_of(num_planes));

        for segment_iter in 0..segments_per_plane {
            let local_segment = segment_iter * num_planes + plane_id;

            let vector = match matrix_layout {
                // mat=rhs
                MatrixLayout::RowMajor => {
                    mat.read_checked((global_k_pos + local_segment, cube_offset))
                }
                // mat=lhs
                MatrixLayout::ColMajor => {
                    mat.read_checked((cube_offset, global_k_pos + local_segment))
                }
            };

            // TODO swizzle
            #[unroll]
            for i in 0..vector_size {
                let row = local_segment;
                let col = local_k_pos + i;
                smem[(row * segment_size + col) as usize] = SM::cast_from(vector[i as usize]);
            }
        }

        sync_cube();

        for segment_iter in 0..segments_per_plane {
            let local_segment = segment_iter * num_planes + plane_id;
            let mut mat_val: Vector<SM, VS> = Vector::empty();
            #[unroll]
            for i in 0..vector_size {
                let row = local_segment;
                let col = local_k_pos + i;

                let transposed_index = col * segment_size + row;
                mat_val[i as usize] = smem[transposed_index as usize];
            }

            accs[segment_iter as usize] += Vector::cast_from(vec_val) * Vector::cast_from(mat_val);
        }

        sync_cube();
    }

    for segment_iter in 0..segments_per_plane {
        let mut sum = AccR::zero();
        let acc = accs[segment_iter as usize];

        #[unroll]
        for i in 0..vector_size {
            sum += acc[i as usize];
        }

        let sum = if comptime!(plane_dim > 1) {
            plane_sum(sum)
        } else {
            sum
        };

        if unit_id == 0 {
            out.write_checked(
                (cube_offset + segment_iter * num_planes + plane_id) as usize,
                O::cast_from(sum),
            );
        }
    }
}
