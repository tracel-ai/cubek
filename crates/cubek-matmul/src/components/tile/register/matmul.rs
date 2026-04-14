use cubecl::prelude::*;
use cubek_std::{
    tile::StridedTile,
    {MatrixLayout, TileSize},
};

use crate::components::tile::{
    TileMatmul, Tilex,
    register::config::RegisterMatmulConfig,
    tilex_allocate, tilex_execute, tilex_load, tilex_write,
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

// #[derive(CubeType)]
// pub struct UnitOperands<L: Numeric, R: Numeric, A: Numeric> {
//     #[cube(comptime)]
//     _phantom: PhantomData<(L, R, A)>,
// }

// impl<L: Numeric, R: Numeric, A: Numeric> Operands for UnitOperands<L, R, A> {
//     type Lhs = UnitFragment<L>;
//     type Rhs = UnitFragment<R>;
//     type Acc = UnitFragment<A>;
// }

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for RegisterMatmul
{
    type Config = RegisterMatmulConfig;

    // TODO dummy
    type Scope = u32;

    fn execute(
        lhs: &Tilex<L, VL, Self::Scope, ReadWrite>,
        rhs: &Tilex<R, VR, Self::Scope, ReadWrite>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_execute(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] _config: Self::Config,
    ) -> Tilex<L, VL, Self::Scope, ReadWrite> {
        tilex_allocate(layout)
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] _config: Self::Config,
    ) -> Tilex<R, VR, Self::Scope, ReadWrite> {
        tilex_allocate(layout)
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] _config: Self::Config,
    ) -> Tilex<A, VA, Self::Scope, ReadWrite> {
        tilex_allocate(layout)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        lhs: &mut Tilex<L, VL, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load(tile, lhs);
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tilex<R, VR, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load(tile, rhs);
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tilex<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_load(tile, acc);
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tilex<E, ES, Self::Scope, ReadWrite>,
        acc: &mut Tilex<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tilex_write(tile, acc);
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
