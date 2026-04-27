use cubecl::{cmma::MmaDefinition, define_size, ir::MatrixIdent, prelude::*};
use cubek_std::{
    MatrixLayout, TileSize,
    tile::mma::{MmaFragmentReader as _, MmaIOConfig, MmaStageReader, MmaStageWriter},
    tile::{Filled, Strided, StridedTile},
};

use crate::components::tile_matmul::{
    MmaAccTile, MmaLhsTile, MmaRhsTile, SharedTileConfig, TileConfig,
};
use crate::components::tile_matmul::{Tile, tile::Scope};

// Fragment inner vector sizes for the three MMA roles. Bound at allocation time
// via `mma_register_vector_sizes` to match the hardware's `def.vector_size(...)`
// for each role — these are independent of the outer Tile enum's stage vector `V`.
define_size!(pub NL);
define_size!(pub NR);
define_size!(pub NA);

#[cube]
fn make_mma_definition<L: Numeric, R: Numeric, A: Numeric>(
    #[comptime] config: SharedTileConfig,
) -> MmaDefinition<L, R, A> {
    MmaDefinition::new(
        config.elements_in_tile_m() as usize,
        config.elements_in_tile_n() as usize,
        config.elements_in_tile_k() as usize,
    )
}

#[cube]
#[allow(unused_variables)]
pub fn mma_register_vector_sizes<L: Numeric, R: Numeric, A: Numeric>(def: MmaDefinition<L, R, A>) {
    let vector_size_a = def.vector_size(MatrixIdent::A);
    let vector_size_b = def.vector_size(MatrixIdent::B);
    let vector_size_acc = def.vector_size(MatrixIdent::Accumulator);
    intrinsic!(|scope| {
        scope.register_size::<NL>(vector_size_a);
        scope.register_size::<NR>(vector_size_b);
        scope.register_size::<NA>(vector_size_acc);
    });
}

// VL/VR/VA in the signatures below are unused for the actual fragment — they exist
// only so the return type matches what the `TileMatmul` trait demands. The fragment
// is sized by NL/NR/NA (hardware-specific register vector size), not by VL/VR/VA.
#[cube]
pub fn mma_allocate_lhs<L: Numeric, VL: Size, R: Numeric, A: Numeric, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
) -> Tile<L, VL, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::A);

    Tile::new_MmaLhs(MmaLhsTile::<L> {
        fragment: Array::new(vector_count),
        matrix_layout: layout,
        config,
        mma_io_config,
    })
}

#[cube]
pub fn mma_allocate_rhs<R: Numeric, VR: Size, L: Numeric, A: Numeric, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
) -> Tile<R, VR, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::B);

    Tile::new_MmaRhs(MmaRhsTile::<R> {
        fragment: Array::new(vector_count),
        matrix_layout: layout,
        config,
        mma_io_config,
    })
}

#[cube]
pub fn mma_allocate_acc<A: Numeric, VA: Size, L: Numeric, R: Numeric, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
) -> Tile<A, VA, Sc, ReadWrite> {
    let def = make_mma_definition::<L, R, A>(config);
    mma_register_vector_sizes(def);
    let vector_count = def.vectors_per_lane(MatrixIdent::Accumulator);

    Tile::new_MmaAcc(MmaAccTile::<A> {
        fragment: Array::new(vector_count),
        matrix_layout: layout,
        config,
        mma_io_config,
    })
}

#[cube]
pub fn mma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NL>>,
    rhs: &Array<Vector<R, NR>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] _matrix_layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] _mma_io_config: MmaIOConfig,
) {
    let def = MmaDefinition::<L, R, A>::new(
        config.elements_in_tile_m() as usize,
        config.elements_in_tile_n() as usize,
        config.elements_in_tile_k() as usize,
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
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
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
            config.elements_in_tile_m(),
            config.elements_in_tile_n(),
            config.elements_in_tile_k(),
        )),
        mma_io_config,
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
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
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
            config.elements_in_tile_m(),
            config.elements_in_tile_n(),
            config.elements_in_tile_k(),
        )),
        mma_io_config,
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
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
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
            config.elements_in_tile_m(),
            config.elements_in_tile_n(),
            config.elements_in_tile_k(),
        )),
        mma_io_config,
    );
}

#[cube]
pub fn mma_load_acc_zeros<E: Numeric, ES: Size, A: Numeric, L: Numeric, R: Numeric>(
    fragment: &mut Array<Vector<A, NA>>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
) {
    let def = make_mma_definition::<L, R, A>(config);
    MmaStageReader::<Filled>::load_fragment::<A, NA, E, ES, L, R, A>(
        &E::from_int(0),
        fragment,
        def,
        MatrixIdent::Accumulator,
        matrix_layout,
        comptime!(TileSize::new(
            config.elements_in_tile_m(),
            config.elements_in_tile_n(),
            config.elements_in_tile_k(),
        )),
        mma_io_config,
    );
}

#[cube]
pub fn mma_write_to_shared<E: Numeric, ES: Size, A: Numeric, L: Numeric, R: Numeric>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    fragment: &Array<Vector<A, NA>>,
    #[comptime] config: SharedTileConfig,
    #[comptime] mma_io_config: MmaIOConfig,
) {
    let def = make_mma_definition::<L, R, A>(config);
    let out_layout = comptime!(shared.layout);
    MmaStageWriter::store_fragment(
        shared,
        fragment,
        def,
        MatrixIdent::Accumulator,
        out_layout,
        config.elements_in_tile_m(),
        mma_io_config,
    );
}
