use cubecl::{cmma::MmaDefinition, ir::MatrixIdent, prelude::*};

use crate::{
    MatrixLayout, TileSize,
    tile::{
        compute::matmul::mma::{MmaFragmentReader as _, MmaStageReader, MmaStageWriter},
        data::{Filled, MmaMatmul, NA, NL, NR, Strided, StridedTile},
    },
};

#[cube]
fn make_mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: MmaMatmul,
) -> MmaDefinition<L, R, A> {
    MmaDefinition::new(
        config.tile_size.m() as usize,
        config.tile_size.n() as usize,
        config.tile_size.k() as usize,
    )
}

#[cube]
pub fn mma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NL>>,
    rhs: &Array<Vector<R, NR>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] _matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let def = MmaDefinition::<L, R, A>::new(
        config.tile_size.m() as usize,
        config.tile_size.n() as usize,
        config.tile_size.k() as usize,
    );
    let out_arr = def.execute(lhs, rhs, acc);
    let num_vectors = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num_vectors {
        acc[i] = out_arr[i];
    }
}

#[cube]
pub fn mma_load_lhs_from_shared<
    E: Numeric,
    ES: Size,
    L: Numeric,
    R: Numeric,
    A: Numeric,
    IO: SliceVisibility,
>(
    shared: &StridedTile<E, ES, IO>,
    fragment: &mut Array<Vector<L, NL>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let shared = shared.to_read_only();
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Strided>::load_fragment(
        &shared,
        fragment,
        def,
        MatrixIdent::A,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_load_rhs_from_shared<
    E: Numeric,
    ES: Size,
    R: Numeric,
    L: Numeric,
    A: Numeric,
    IO: SliceVisibility,
>(
    shared: &StridedTile<E, ES, IO>,
    fragment: &mut Array<Vector<R, NR>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let shared = shared.to_read_only();
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Strided>::load_fragment(
        &shared,
        fragment,
        def,
        MatrixIdent::B,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_load_acc_from_shared<
    E: Numeric,
    ES: Size,
    A: Numeric,
    L: Numeric,
    R: Numeric,
    IO: SliceVisibility,
>(
    shared: &StridedTile<E, ES, IO>,
    fragment: &mut Array<Vector<A, NA>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let shared = shared.to_read_only();
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Strided>::load_fragment(
        &shared,
        fragment,
        def,
        MatrixIdent::Accumulator,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_load_acc_zeros<E: Numeric, ES: Size, A: Numeric, L: Numeric, R: Numeric>(
    fragment: &mut Array<Vector<A, NA>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: MmaMatmul,
) {
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Filled>::load_fragment::<A, NA, E, ES, L, R, A>(
        &E::from_int(0),
        fragment,
        def,
        MatrixIdent::Accumulator,
        matrix_layout,
        comptime!(TileSize::new(
            config.tile_size.m(),
            config.tile_size.n(),
            config.tile_size.k(),
        )),
        config.mma_io_config,
    );
}

#[cube]
pub fn mma_write_to_shared<E: Numeric, ES: Size, A: Numeric, L: Numeric, R: Numeric>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    fragment: &Array<Vector<A, NA>>,
    #[comptime] config: MmaMatmul,
) {
    let def = make_mma_definition::<L, R, A>(config);
    let out_layout = comptime!(shared.layout);
    MmaStageWriter::store_fragment(
        shared,
        fragment,
        def,
        MatrixIdent::Accumulator,
        out_layout,
        config.tile_size.m(),
        config.mma_io_config,
    );
}
