use std::marker::PhantomData;

use crate::{
    components::batch::{
        BatchConfig as _, BatchMatmul, CheckBounds, SliceIndex,
        base::BatchMatmulFamily,
        gemv_unit_perpendicular::{
            VecMatUnitPerpendicularBlueprint, VecMatUnitPerpendicularConfig,
            VecMatUnitPerpendicularFamily,
        },
    },
    definition::{cube_pos_to_matrix_batch, *},
    launch::MatmulArgs,
};

use cubecl::{
    prelude::*,
    {cube, num_traits::Zero},
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
    #[comptime] blueprint: VecMatUnitPerpendicularBlueprint,
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
    let config = comptime!(VecMatUnitPerpendicularFamily::expand_config(
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

    VecMatUnitPerpendicular::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs, LhsSize),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs, RhsSize),
        (Acc, AccSize, Acc, AccSize, RegisterAcc, AccSize),
    )>::execute::<Args>(&mut state, cube_mapping, config);
}

pub struct VecMatUnitPerpendicular<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for VecMatUnitPerpendicular<MP> {
    type Config = VecMatUnitPerpendicularConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let num_planes = config.num_planes;
        let plane_dim = config.plane_dim;
        let check_bounds = config.check_bounds;

        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        let (_, _, k) = lhs.shape();
        let (_, _, n) = out.shape();
        let (n_cube_id, batch_cube_id) = cube_pos_to_matrix_batch(&cube_mapping);

        let lhs_batch = Args::batch_lhs(state, batch_cube_id as usize);
        let rhs_batch = Args::batch_rhs(state, batch_cube_id as usize);
        let out_batch = Args::batch_out(state, batch_cube_id as usize);

        let lhs = lhs.view(SliceIndex::new(lhs_batch, lhs.shape()));
        let rhs = rhs.view(SliceIndex::new(rhs_batch, rhs.shape()));
        let out = out.view_mut(SliceIndex::new(out_batch, out.shape()));

        let size!(NA) = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let vector_size = NA::value() as u32;

        let plane_id = UNIT_POS_Y;
        let unit_id = UNIT_POS_X;

        let tile_size = plane_dim * vector_size;
        let absolute_plane_id = n_cube_id * num_planes + plane_id;
        let unit_pos_n = absolute_plane_id * plane_dim + unit_id;
        let vectorized_pos_n = unit_pos_n * vector_size;

        // The first if statement is running at comptime.
        if comptime!(matches!(check_bounds, CheckBounds::Terminate)) {
            // This is a runtime cond.
            let should_terminate = vectorized_pos_n >= n;
            if should_terminate {
                terminate!();
            }
        }

        let num_tiles = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            k.div_ceil(tile_size)
        } else {
            k / tile_size
        };

        let mut acc = Vector::<AccRE<MP>, NA>::zero();

        for tile_index in 0..num_tiles {
            let swizzled_tile_index = (tile_index + plane_id) % num_tiles;
            let k_base = swizzled_tile_index * plane_dim;

            let local_lhs_vec = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
                lhs.read_checked((0, (k_base + unit_id) * vector_size))
            } else {
                lhs.read_unchecked((0, (k_base + unit_id) * vector_size))
            };

            for plane_iter in 0..plane_dim {
                let lhs_vec = shuffle(local_lhs_vec, plane_iter, plane_dim);
                let rhs_k_vec_base = (k_base + plane_iter) * vector_size;

                for vec_iter in 0..NA::value() as u32 {
                    let lhs_scalar = lhs_vec[vec_iter as usize];
                    let rhs_vec = if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
                        rhs.read_checked((rhs_k_vec_base + vec_iter, vectorized_pos_n))
                    } else {
                        rhs.read_unchecked((rhs_k_vec_base + vec_iter, vectorized_pos_n))
                    };
                    acc += Vector::cast_from(lhs_scalar) * Vector::cast_from(rhs_vec);
                }
            }
        }

        if comptime!(matches!(check_bounds, CheckBounds::Checked)) {
            out.write_checked((0, vectorized_pos_n), Vector::cast_from(acc));
        } else {
            out.write((0, vectorized_pos_n), Vector::cast_from(acc));
        }
    }
}

#[cube]
fn shuffle<E: Numeric, N: Size>(
    shared_value: Vector<E, N>,
    unit: u32,
    #[comptime] plane_dim: u32,
) -> Vector<E, N> {
    if comptime!(plane_dim > 1) {
        plane_shuffle(shared_value, unit)
    } else {
        shared_value
    }
}
