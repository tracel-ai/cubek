use cubecl::{define_size, prelude::*};
use cubek_std::{
    MatrixLayout,
    tile::{
        Strided, StridedTile,
        mma::{MmaFragmentReader, MmaStageReader, MmaStageWriter},
    },
};

use crate::components::tile::{StandardTileIO, TileMatmul, mma::config::MmaMatmulConfig};
use cubecl::{cmma::MmaDefinition, ir::MatrixIdent};

/// Uses one plane to perform a small matmul using accelerated instructions, with manual register
/// management.
/// Currently requires matrix layout to match the platform's preferred layout.
pub struct MmaMatmul {}

define_size!(pub NL);
define_size!(pub NR);
define_size!(pub NA);

#[derive(CubeType)]
pub struct MmaFragment<E: Numeric, N: Size> {
    fragment: Array<Vector<E, N>>,
    #[cube(comptime)]
    layout: MatrixLayout,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for MmaMatmul
where
    MmaStageReader<Strided>: MmaFragmentReader<TileKind = Strided>,
    MmaStageReader<Strided>: MmaFragmentReader<TileKind = Strided>,
    MmaStageReader<Option<Strided>>: MmaFragmentReader<TileKind = Option<Strided>>,
{
    type Config = MmaMatmulConfig;

    type LhsFragment = MmaFragment<L, NL>;
    type RhsFragment = MmaFragment<R, NR>;
    type AccFragment = MmaFragment<A, NA>;

    type TileIO = StandardTileIO;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        out: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        let def = mma_definition(config);
        let out_arr = def.execute(&lhs.fragment, &rhs.fragment, &out.fragment);
        let num_vectors = def.vectors_per_lane(MatrixIdent::Accumulator);

        #[unroll]
        for i in 0..num_vectors {
            out.fragment[i] = out_arr[i];
        }
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        let def = mma_definition::<L, R, A>(config);
        register_vector_sizes(def);
        let vector_count = def.vectors_per_lane(MatrixIdent::A);

        MmaFragment::<L, NL> {
            fragment: Array::new(vector_count),
            layout,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        let def = mma_definition::<L, R, A>(config);
        register_vector_sizes(def);
        let vector_count = def.vectors_per_lane(MatrixIdent::B);

        MmaFragment::<R, NR> {
            fragment: Array::new(vector_count),
            layout,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        let def = mma_definition::<L, R, A>(config);
        register_vector_sizes(def);
        let vector_count = def.vectors_per_lane(MatrixIdent::Accumulator);

        MmaFragment::<A, NA> {
            fragment: Array::new(vector_count),
            layout,
        }
    }

    fn load_lhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Strided>::load_fragment(
            tile,
            &mut lhs.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::A,
            lhs.layout,
            config.shared.tile_size,
            config.mma_io_config,
        );
    }

    fn load_rhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Strided>::load_fragment(
            tile,
            &mut rhs.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::B,
            rhs.layout,
            config.shared.tile_size,
            config.mma_io_config,
        );
    }

    fn load_acc<E: Numeric, N: Size>(
        tile: &ComptimeOption<StridedTile<E, N>>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageReader::<Option<Strided>>::load_fragment(
            tile,
            &mut acc.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            acc.layout,
            config.shared.tile_size,
            config.mma_io_config,
        );
    }

    fn write_results<E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        out: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        MmaStageWriter::store_fragment(
            tile,
            &out.fragment,
            mma_definition::<L, R, A>(config),
            MatrixIdent::Accumulator,
            tile.layout,
            config.shared.tile_size.m(),
            config.mma_io_config,
        );
    }
}

#[cube]
pub(super) fn mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: MmaMatmulConfig,
) -> MmaDefinition<L, R, A> {
    let size = config.shared.tile_size;
    MmaDefinition::new(size.m() as usize, size.n() as usize, size.k() as usize)
}

#[cube]
#[allow(unused_variables)]
pub(super) fn register_vector_sizes<L: Numeric, R: Numeric, A: Numeric>(
    def: MmaDefinition<L, R, A>,
) {
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_size_acc = def.vector_size(MatrixIdent::Accumulator);
    intrinsic!(|scope| {
        scope.register_size::<NL>(vector_size_a);
        scope.register_size::<NR>(vector_size_b);
        scope.register_size::<NA>(vector_size_acc);
    });
}
