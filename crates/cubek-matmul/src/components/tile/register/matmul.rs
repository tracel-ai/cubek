use cubecl::prelude::*;
use cubek_std::{
    tile::{Filled, Strided, StridedTile},
    {MatrixLayout, TileSize},
};

use crate::{
    components::tile::{
        StandardTileIO, TileMatmul,
        register::{
            config::{ProductType, RegisterMatmulConfig},
            reader::{RegisterFragmentReader, RegisterStageReader},
            writer::RegisterStageWriter,
        },
    },
    definition::StageIdent,
};

/// Uses one unit to perform a small matmul directly in registers
pub struct RegisterMatmul {}

/// Doesn't impact performance much, but may increase kernel size too much when true (often ~6X).
///
/// TODO: make it configurable
pub(super) const UNROLL: bool = false;

#[derive(CubeType)]
pub struct UnitFragment<E: Numeric> {
    pub array: Array<E>,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

#[cube]
impl<L: Numeric, R: Numeric, A: Numeric> TileMatmul<L, R, A> for RegisterMatmul
where
    RegisterStageReader<Filled>: RegisterFragmentReader<TileKind = Filled>,
{
    type Config = RegisterMatmulConfig;

    type LhsFragment = UnitFragment<L>;
    type RhsFragment = UnitFragment<R>;
    type AccFragment = UnitFragment<A>;

    type TileIO = StandardTileIO;

    fn execute(
        lhs: &Self::LhsFragment,
        rhs: &Self::RhsFragment,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        match config.product_type {
            ProductType::Inner => Self::inner_product(
                &lhs.array,
                &rhs.array,
                &mut acc.array,
                config.shared.tile_size,
            ),
            ProductType::Outer => Self::outer_product(
                &lhs.array,
                &rhs.array,
                &mut acc.array,
                config.shared.tile_size,
            ),
        }
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::LhsFragment {
        UnitFragment::<L> {
            array: Array::new(config.shared.tile_size.mk() as usize),
            layout,
        }
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::RhsFragment {
        UnitFragment::<R> {
            array: Array::new(config.shared.tile_size.nk() as usize),
            layout,
        }
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Self::AccFragment {
        UnitFragment::<A> {
            array: Array::new(config.shared.tile_size.mn() as usize),
            layout,
        }
    }

    fn load_lhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        lhs: &mut Self::LhsFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageReader::<Strided>::load_fragment(tile, lhs, StageIdent::Lhs, config)
    }

    fn load_rhs<E: Numeric, N: Size>(
        tile: &StridedTile<E, N>,
        rhs: &mut Self::RhsFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageReader::<Strided>::load_fragment(tile, rhs, StageIdent::Rhs, config)
    }

    fn load_acc<E: Numeric, N: Size>(
        tile: &ComptimeOption<StridedTile<E, N>>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageReader::<Option<Strided>>::load_fragment(tile, acc, StageIdent::Acc, config);
    }

    fn write_results<E: Numeric, N: Size>(
        tile: &mut StridedTile<E, N, ReadWrite>,
        acc: &mut Self::AccFragment,
        #[comptime] config: Self::Config,
    ) {
        RegisterStageWriter::store_fragment(tile, acc, config)
    }
}

#[cube]
impl RegisterMatmul {
    pub fn inner_product<Lhs: Numeric, Rhs: Numeric, EA: Numeric>(
        lhs: &Array<Lhs>,
        rhs: &Array<Rhs>,
        acc: &mut Array<EA>,
        #[comptime] tile_size: TileSize,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = tile_size.into(); (m, n, k)};

        #[unroll(UNROLL)]
        for m_ in 0..m {
            #[unroll(UNROLL)]
            for n_ in 0..n {
                #[unroll(UNROLL)]
                for k_ in 0..k {
                    let lhs_elem = EA::cast_from(lhs[(m_ * k + k_) as usize]);
                    let rhs_elem = EA::cast_from(rhs[(n_ * k + k_) as usize]);
                    acc[(m_ * n + n_) as usize] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    pub fn outer_product<Lhs: Numeric, Rhs: Numeric, EA: Numeric>(
        lhs: &Array<Lhs>,
        rhs: &Array<Rhs>,
        acc: &mut Array<EA>,
        #[comptime] tile_size: TileSize,
    ) {
        let (m, n, k) = comptime! {let (m, n, k): (u32, u32, u32) = tile_size.into(); (m, n, k)};

        #[unroll(UNROLL)]
        for k_ in 0..k {
            #[unroll(UNROLL)]
            for m_ in 0..m {
                let lhs_elem = EA::cast_from(lhs[(k_ * m + m_) as usize]);
                #[unroll(UNROLL)]
                for n_ in 0..n {
                    let rhs_elem = EA::cast_from(rhs[(k_ * n + n_) as usize]);
                    acc[(m_ * n + n_) as usize] += lhs_elem * rhs_elem;
                }
            }
        }
    }

    pub fn load_plain<ES: Numeric, NS: Size, ER: Numeric>(
        tile: &StridedTile<ES, NS>,
        array: &mut Array<ER>,
        #[comptime] num_segments: u32,
        #[comptime] segment_size: u32,
    ) {
        let vector_size = NS::value().comptime() as u32;
        let num_vectors_per_segment = segment_size / vector_size;

        #[unroll(UNROLL)]
        for segment in 0..num_segments {
            #[unroll(UNROLL)]
            for vector_within_segment in 0..num_vectors_per_segment {
                let vector = tile.get_vector(segment, vector_within_segment);
                #[unroll]
                for pos_within_vector in 0..vector_size {
                    let offs = segment * segment_size
                        + vector_within_segment * vector_size
                        + pos_within_vector;
                    array[offs as usize] = ER::cast_from(vector[pos_within_vector as usize]);
                }
            }
        }
    }

    pub fn load_transposed<ES: Numeric, NS: Size, ER: Numeric>(
        tile: &StridedTile<ES, NS>,
        array: &mut Array<ER>,
        #[comptime] num_segments: u32,
        #[comptime] segment_size: u32,
    ) {
        let vector_size = NS::value().comptime() as u32;
        let num_vectors_per_segment = segment_size / vector_size;

        #[unroll(UNROLL)]
        for segment in 0..num_segments {
            #[unroll(UNROLL)]
            for vector_within_segment in 0..num_vectors_per_segment {
                let vector = tile.get_vector(segment, vector_within_segment);
                #[unroll]
                for pos_within_vector in 0..vector_size {
                    let offs = (vector_within_segment * vector_size + pos_within_vector)
                        * num_segments
                        + segment;
                    array[offs as usize] = ER::cast_from(vector[pos_within_vector as usize]);
                }
            }
        }
    }
}
