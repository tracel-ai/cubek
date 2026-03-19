use std::marker::PhantomData;

use crate::components::batch::base::BatchMatmulFamily;
use crate::components::batch::vec2mat::{Vec2MatBlueprint, Vec2MatFamily, Vec2MatMatmulConfig};
use crate::components::batch::{BatchConfig as _, SliceIndex};

use crate::{components::batch::BatchMatmul, definition::*, launch::MatmulArgs};
use cubecl::prelude::*;
use cubecl::std::tensor::View;
use cubecl::std::tensor::layout::Coords2d;
use cubecl::{cube, num_traits::Zero};
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
        _cube_mapping: CubeMapping,
        #[comptime] config: Self::Config,
    ) {
        let lhs = Args::view_lhs(state);
        let rhs = Args::view_rhs(state);
        let out = Args::view_out(state);

        let (_, _, k) = lhs.shape();
        // m=1
        let (_, _, n) = out.shape();

        let lhs = lhs.view(SliceIndex::new(0, lhs.shape()));
        let rhs = rhs.view(SliceIndex::new(0, rhs.shape()));
        let mut out = out.view_mut(SliceIndex::new(0, out.shape()));

        // if  n >= size_n {
        //     terminate!();
        // }

        let vector_size = comptime![Ord::max(lhs.vector_size(), rhs.vector_size())];
        let size!(NA) = vector_size;
        let mut sum = Vector::<AccR<MP>, NA>::zero();

        // Assuming CUBE_DIM_X = plane size
        let plane_offset = UNIT_POS_Y * CUBE_DIM_X;
        let unit_offset = UNIT_POS_PLANE;

        let mat_index_n = plane_offset + unit_offset;

        // for k in range_stepped(0u32, k, vector_size as u32) {
        let num_tiles = k / CUBE_DIM_X;
        for tile_i in 0..num_tiles {
            let swizzled_tile_i = (tile_i * CUBE_DIM_X + UNIT_POS_Y) % k;

            let vec_index = swizzled_tile_i * CUBE_DIM_X + unit_offset;
            let vecval = lhs.read((0, vec_index));

            #[unroll]
            for x in 0..32u32 {
                let xth_val = plane_broadcast(vecval, x);

                let mat_index_k = vec_index;
                let matval = rhs.read((mat_index_k, mat_index_n));
                sum += Vector::cast_from(xth_val) * Vector::cast_from(matval);
            }
        }

        out.write((0, mat_index_n), Vector::cast_from(sum));
    }
}

#[cube]
fn load_unrolled<I: Numeric, N: Size, N2: Size>(
    view: &View<Vector<I, N>, Coords2d>,
    pos: Coords2d,
    #[comptime] layout: MatrixLayout,
) -> Vector<I, N2> {
    let vector_size = N2::value();
    comptime![assert!(vector_size >= view.vector_size())];
    let view_vector_size = view.vector_size();
    if view.vector_size().comptime() == vector_size {
        Vector::cast_from(view[pos])
    } else {
        let (row, col) = pos;
        let mut out = Vector::empty();
        #[unroll]
        for i in range_stepped(0, vector_size as u32, view_vector_size as u32) {
            let pos = match layout {
                MatrixLayout::RowMajor => (row, col + i),
                MatrixLayout::ColMajor => (row + i, col),
            };
            let value = view[pos];
            #[unroll]
            for n in 0..view_vector_size {
                out[i as usize + n] = value[n];
            }
        }
        out
    }
}
