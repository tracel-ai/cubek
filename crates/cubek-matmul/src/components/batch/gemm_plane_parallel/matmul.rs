use std::marker::PhantomData;

use crate::components::batch::{
    BatchConfig as _, BatchMatmul, BatchMatmulFamily, CheckBounds,
    gemm_plane_parallel::{
        GemmPlaneParallelBlueprint, GemmPlaneParallelConfig, GemmPlaneParallelFamily,
        layout::MatLayout,
    },
};

use crate::{
    definition::{cube_pos_to_m_n_batch, *},
    launch::MatmulArgs,
};
use cubecl::{cube, num_traits::Zero, std::tensor::layout::Coords2d};
use cubecl::{prelude::*, std::tensor::View};

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
        let (m_cube, n_cube, batch_cube) = cube_pos_to_m_n_batch(&cube_mapping);

        let lhs_batch = Args::batch_lhs(state, batch_cube as usize);
        let rhs_batch = Args::batch_rhs(state, batch_cube as usize);
        let out_batch = Args::batch_out(state, batch_cube as usize);

        let vector_size = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let size!(N) = vector_size;

        let check_bounds = config.check_bounds;

        execute_gemm::<LhsG<MP>, RhsG<MP>, AccG<MP>, AccRE<MP>, N>(
            lhs.view(MatLayout::new(lhs_batch, (m, k))),
            rhs.view(MatLayout::new(rhs_batch, (k, n))),
            out.view_mut(MatLayout::new(out_batch, (m, n))),
            m_cube,
            n_cube,
            k,
            config.num_planes,
            config.plane_dim,
            vector_size as u32,
            check_bounds,
        );
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn execute_gemm<L: CubePrimitive, R: CubePrimitive, O: CubePrimitive, AccR: Numeric, N: Size>(
    lhs: View<L, Coords2d>,
    rhs: View<R, Coords2d>,
    out: View<O, Coords2d, ReadWrite>,
    m_cube: u32,
    n_cube: u32,
    k_dim: u32,
    #[comptime] num_planes: u32,
    #[comptime] plane_dim: u32,
    #[comptime] vector_size: u32,
    #[comptime] check_bounds: CheckBounds,
) {
    let plane_id = UNIT_POS_Y;
    let unit_id = UNIT_POS_X;

    let (out_m, out_n) = out.shape();

    // One plane per output column within a cube (similar to gemv VecMatColMajor),
    // looping over m rows inside the kernel.
    let n_pos = n_cube * num_planes + plane_id;

    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let should_terminate = n_pos >= out_n;
        if should_terminate {
            terminate!();
        }
    }

    let segment_size = plane_dim * vector_size;
    let num_segments_k = k_dim / segment_size;

    // Each cube currently owns a single m row; m_cube enumerates rows.
    let m_pos = m_cube;
    if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
        let should_terminate = m_pos >= out_m;
        if should_terminate {
            terminate!();
        }
    }

    let mut acc = Vector::<AccR, N>::zero();

    for segment_index in 0..num_segments_k {
        let swizzled_segment_index = (segment_index + plane_id) % num_segments_k;
        let k_base = swizzled_segment_index * plane_dim;

        let k_pos = (k_base + unit_id) * vector_size;

        let lhs_val = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            lhs.read_checked((m_pos, k_pos))
        } else {
            lhs.read_unchecked((m_pos, k_pos))
        };

        let rhs_val = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            rhs.read_checked((k_pos, n_pos))
        } else {
            rhs.read_unchecked((k_pos, n_pos))
        };

        acc += Vector::cast_from(lhs_val) * Vector::cast_from(rhs_val);
    }

    let sum = Vector::vector_sum(acc);

    let sum = if comptime!(plane_dim > 1) {
        O::cast_from(plane_sum(sum))
    } else {
        O::cast_from(sum)
    };

    #[allow(clippy::collapsible_else_if)]
    if comptime!(plane_dim == 1) {
        if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            out.write_checked((m_pos, n_pos), sum);
        } else {
            out.write((m_pos, n_pos), sum);
        }
    } else {
        if unit_id == 0 {
            if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
                out.write_checked((m_pos, n_pos), sum);
            } else {
                out.write((m_pos, n_pos), sum);
            }
        }
    }
}
