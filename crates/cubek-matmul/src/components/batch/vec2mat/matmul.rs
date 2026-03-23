use std::marker::PhantomData;

use crate::components::batch::base::BatchMatmulFamily;
use crate::components::batch::vec2mat::{Vec2MatBlueprint, Vec2MatFamily, Vec2MatMatmulConfig};
use crate::components::batch::{BatchConfig as _, SliceIndex};

use crate::{components::batch::BatchMatmul, definition::*, launch::MatmulArgs};
use cubecl::prelude::*;
use cubecl::{cube, num_traits::Zero};

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
    #[comptime] blueprint: Vec2MatBlueprint,
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
    let config = comptime!(Vec2MatFamily::expand_config(
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

    Vec2Mat::<(
        (Lhs, LhsSize, Lhs, LhsSize, RegisterLhs),
        (Rhs, RhsSize, Rhs, RhsSize, RegisterRhs),
        (Acc, AccSize, Acc, AccSize, RegisterAcc),
    )>::execute::<Args>(&mut state, cube_mapping, config);
}

pub struct Vec2Mat<MP: MatmulTypes> {
    _phantom: PhantomData<MP>,
}

#[cube]
impl<MP: MatmulTypes> BatchMatmul<(), MP> for Vec2Mat<MP> {
    type Config = Vec2MatMatmulConfig;

    fn execute<Args: MatmulArgs>(
        state: &mut Args::State<LhsG<MP>, RhsG<MP>, AccG<MP>>,
        cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let num_planes = config.num_planes;
        let plane_dim = config.plane_dim;
        let (_, n_index, _) = cube_mapping.cube_pos_to_tensor_pos();
        // m_index should be 1
        // batch_index: not supported yet

        let plane_id = UNIT_POS_Y;
        let unit_id = UNIT_POS_X;

        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        // m=1
        // let (_, _, n) = out.shape();
        let (_, _, k) = lhs.shape();

        let lhs = lhs.view(SliceIndex::new(0, lhs.shape()));
        let rhs = rhs.view(SliceIndex::new(0, rhs.shape()));
        let out = out.view_mut(SliceIndex::new(0, out.shape()));

        let size!(NA) = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];

        let tile_size = plane_dim * NA::value() as u32;
        let cube_offset = n_index * num_planes * tile_size;
        let plane_offset = plane_id * tile_size;
        let n_pos = cube_offset + plane_offset + unit_id;

        let num_tiles = k / tile_size;

        let mut acc = Vector::<AccR<MP>, NA>::zero();

        for tile_index in 0..num_tiles {
            let swizzled_tile_index = (tile_index + plane_id) % num_tiles;
            let k_base = swizzled_tile_index * plane_dim;

            // Load the whole lhs tile
            let local_lhs_vec = lhs.read((0, k_base + unit_id));

            #[unroll]
            for plane_iter in 0..plane_dim {
                let lhs_vec = if comptime!(plane_dim > 1) {
                    plane_broadcast(local_lhs_vec, plane_iter)
                } else {
                    local_lhs_vec
                };

                for vec_iter in 0..NA::value() as u32 {
                    let lhs_scalar = lhs_vec[vec_iter as usize];
                    let rhs_vec = rhs.read((k_base + plane_iter + vec_iter, n_pos));
                    acc += Vector::cast_from(lhs_scalar) * Vector::cast_from(rhs_vec);
                }
            }
        }

        out.write((0, n_pos), Vector::cast_from(acc));
    }
}
