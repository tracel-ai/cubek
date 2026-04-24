use std::marker::PhantomData;

use cubecl::prelude::*;
use cubek_std::{MatrixLayout, tile::StridedTile};

use crate::components::tile_matmul::tile::Scope;
use crate::components::tile_matmul::{SharedTileConfig, TileConfig};
use crate::definition::StageIdent;

use super::{NPlaneVec, PlaneVecTile, Tile};

// Binds the plane_vec_mat fragment's inner vector size (`NPlaneVec`) to the
// `reduce_vector_size` chosen by the tile config at allocation time.
#[cube]
#[allow(unused_variables)]
fn register_reduce_vector_size(#[comptime] reduce_vector_size: u32) {
    intrinsic!(|scope| {
        scope.register_size::<NPlaneVec>(reduce_vector_size as usize);
    });
}

// ===========================================================================
// Allocate
// ===========================================================================

#[cube]
pub fn planevec_allocate_lhs<L: Numeric, VL: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] reduce_vector_size: u32,
) -> Tile<L, VL, Sc, ReadWrite> {
    register_reduce_vector_size(reduce_vector_size);
    Tile::new_PlaneVec(PlaneVecTile::<L, VL> {
        data: Array::new(1usize),
        matrix_layout: layout,
        config,
        reduce_vector_size,
        _phantom_v: PhantomData,
    })
}

#[cube]
pub fn planevec_allocate_rhs<R: Numeric, VR: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] reduce_vector_size: u32,
) -> Tile<R, VR, Sc, ReadWrite> {
    register_reduce_vector_size(reduce_vector_size);
    Tile::new_PlaneVec(PlaneVecTile::<R, VR> {
        data: Array::new(config.elements_in_tile_n() as usize),
        matrix_layout: layout,
        config,
        reduce_vector_size,
        _phantom_v: PhantomData,
    })
}

#[cube]
pub fn planevec_allocate_acc<A: Numeric, VA: Size, Sc: Scope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: SharedTileConfig,
    #[comptime] reduce_vector_size: u32,
) -> Tile<A, VA, Sc, ReadWrite> {
    register_reduce_vector_size(reduce_vector_size);
    Tile::new_PlaneVec(PlaneVecTile::<A, VA> {
        data: Array::new(config.elements_in_tile_n() as usize),
        matrix_layout: layout,
        config,
        reduce_vector_size,
        _phantom_v: PhantomData,
    })
}

// ===========================================================================
// Execute: (PlaneVec, PlaneVec, PlaneVec)
// ===========================================================================

#[cube]
pub fn planevec_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NPlaneVec>>,
    rhs: &Array<Vector<R, NPlaneVec>>,
    acc: &mut Array<Vector<A, NPlaneVec>>,
    #[comptime] config: SharedTileConfig,
) {
    let n = config.elements_in_tile_n();
    #[unroll]
    for n_idx in 0..n as usize {
        let mut acc_vec = acc[n_idx];
        #[unroll]
        for vi in 0..NPlaneVec::value() {
            let lhs_elem = A::cast_from(lhs[0usize][vi]);
            let rhs_elem = A::cast_from(rhs[n_idx][vi]);
            acc_vec[vi] += plane_sum(lhs_elem * rhs_elem);
        }
        acc[n_idx] = acc_vec;
    }
}

// ===========================================================================
// Load: SharedMemory -> PlaneVec
// ===========================================================================

#[cube]
pub fn planevec_load_from_shared<E: Numeric, ES: Size, N: Numeric, IO: SliceVisibility>(
    shared: &StridedTile<E, ES, IO>,
    arr: &mut Array<Vector<N, NPlaneVec>>,
    #[comptime] config: SharedTileConfig,
    #[comptime] ident: StageIdent,
) {
    match ident {
        StageIdent::Lhs => {
            let offset = shared.stage_offset(UNIT_POS_X);
            arr[0usize] = Vector::cast_from(shared.container[offset as usize]);
        }
        StageIdent::Rhs | StageIdent::Acc => {
            let n = config.elements_in_tile_n();
            #[unroll]
            for n_idx in 0..n {
                let offset = shared.stage_offset(UNIT_POS_X + n_idx * shared.stride);
                arr[n_idx as usize] = Vector::cast_from(shared.container[offset as usize]);
            }
        }
        _ => panic!("Invalid ident for PlaneVec load"),
    }
}

// ===========================================================================
// Load: None -> PlaneVec (zero fill)
// ===========================================================================

#[cube]
pub fn planevec_load_zeros<N: Numeric>(
    arr: &mut Array<Vector<N, NPlaneVec>>,
    #[comptime] config: SharedTileConfig,
) {
    let n = config.elements_in_tile_n();
    let zero = N::from_int(0);
    #[unroll]
    for n_idx in 0..n as usize {
        arr[n_idx] = Vector::cast_from(zero);
    }
}

// ===========================================================================
// Write: PlaneVec -> SharedMemory
// ===========================================================================

#[cube]
pub fn planevec_write_to_shared<A: Numeric, E: Numeric, ES: Size>(
    shared: &mut StridedTile<E, ES, ReadWrite>,
    arr: &Array<Vector<A, NPlaneVec>>,
    #[comptime] config: SharedTileConfig,
    #[comptime] reduce_vector_size: u32,
) {
    if UNIT_POS_X == 0 {
        let out_vector_size = shared.container.vector_size().comptime();
        let n = config.elements_in_tile_n();
        let total_out_vectors = n as usize / out_vector_size;
        let reduce_vec = reduce_vector_size as usize;

        #[unroll]
        for out_vector_iter in 0..total_out_vectors {
            let mut out_vector = Vector::<E, ES>::empty();
            #[unroll]
            for within_vector in 0..out_vector_size {
                let n_idx = out_vector_iter * out_vector_size + within_vector;
                let acc_vec = arr[n_idx];
                let mut sum = A::from_int(0);
                for i in 0..reduce_vec {
                    sum += acc_vec[i];
                }
                out_vector[within_vector] = E::cast_from(sum);
            }
            let offset = shared.stage_offset(out_vector_iter as u32);
            shared.container[offset as usize] = out_vector;
        }
    }
}
