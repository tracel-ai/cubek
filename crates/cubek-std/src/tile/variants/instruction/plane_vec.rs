use cubecl::{define_size, prelude::*};

use crate::{
    MatrixLayout, StageIdent, TileSize,
    tile::{SharedTile, Tile, TileKind, TileKindExpand, TileScope},
};

// plane_vec_mat's fragment inner vector size (= reduce_vector_size). Bound at
// allocate time via `scope.register_size::<NPlaneVec>(reduce_vector_size)`.
// Decoupled from the outer enum `V` so the fragment is sized by the tile impl's
// needs, not the stage's vector size.
define_size!(pub NPlaneVec);

/// Plane-vec tile. Holds the per-unit fragment plus the minimal comptime data
/// the tile body actually uses (`tile_size` for the n iteration, plus the
/// register-size hookup info implicit in the [`NPlaneVec`] binding done at
/// allocation time). The matmul-level configuration that produced these
/// values lives in cubek-matmul as `PlaneVecMatInnerProduct`.
#[derive(CubeType)]
pub struct PlaneVecTile<N: Numeric> {
    // Fragment inner size is `NPlaneVec` (= reduce_vector_size).
    pub data: Array<Vector<N, NPlaneVec>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub tile_size: TileSize,
    /// Inner reduction vector size for `NPlaneVec`. Carried because
    /// `planevec_write_to_shared` needs the extent at use-site (`NPlaneVec::value()`
    /// isn't observable from a `#[cube]` callsite).
    #[cube(comptime)]
    pub reduce_vector_size: u32,
}

// Binds the plane_vec_mat fragment's inner vector size (`NPlaneVec`) to the
// `reduce_vector_size` chosen by the tile config at allocation time.
#[cube]
#[allow(unused_variables)]
fn register_reduce_vector_size(#[comptime] reduce_vector_size: u32) {
    intrinsic!(|scope| {
        scope.register_size::<NPlaneVec>(reduce_vector_size as usize);
    });
}

#[cube]
pub fn planevec_allocate_lhs<L: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] reduce_vector_size: u32,
) -> Tile<L, Sc> {
    register_reduce_vector_size(reduce_vector_size);
    Tile::from_kind(TileKind::new_PlaneVec(PlaneVecTile::<L> {
        data: Array::new(1usize),
        matrix_layout: layout,
        tile_size,
        reduce_vector_size,
    }))
}

#[cube]
pub fn planevec_allocate_rhs<R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] reduce_vector_size: u32,
) -> Tile<R, Sc> {
    register_reduce_vector_size(reduce_vector_size);
    Tile::from_kind(TileKind::new_PlaneVec(PlaneVecTile::<R> {
        data: Array::new(tile_size.n() as usize),
        matrix_layout: layout,
        tile_size,
        reduce_vector_size,
    }))
}

#[cube]
pub fn planevec_allocate_acc<A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] reduce_vector_size: u32,
) -> Tile<A, Sc> {
    register_reduce_vector_size(reduce_vector_size);
    Tile::from_kind(TileKind::new_PlaneVec(PlaneVecTile::<A> {
        data: Array::new(tile_size.n() as usize),
        matrix_layout: layout,
        tile_size,
        reduce_vector_size,
    }))
}

#[cube]
impl<A: Numeric> PlaneVecTile<A> {
    /// Executes `lhs · rhs`, accumulating into `self` via the plane-vec
    /// inner-product matmul.
    pub fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &PlaneVecTile<L>, rhs: &PlaneVecTile<R>) {
        planevec_execute(&lhs.data, &rhs.data, &mut self.data, self.tile_size);
    }
}

#[cube]
impl<N: Numeric> PlaneVecTile<N> {
    /// Copies into the plane-vec tile from `source`. Supported sources:
    /// `SharedMemory` and `None` (zero-init).
    pub fn copy_from<SE: Numeric, SS: Size, Sc: TileScope>(
        &mut self,
        source: &Tile<SE, Sc>,
        #[comptime] ident: StageIdent,
    ) {
        match &source.kind {
            TileKind::SharedTile(shared) => {
                planevec_load_from_shared::<SE, SS, N>(
                    shared,
                    &mut self.data,
                    self.tile_size,
                    ident,
                );
            }
            TileKind::None => planevec_load_zeros::<N>(&mut self.data, self.tile_size),
            TileKind::Cmma(_)
            | TileKind::Mma(_)
            | TileKind::Register(_)
            | TileKind::PlaneVec(_)
            | TileKind::Interleaved(_)
            | TileKind::Unit(_)
            | TileKind::WhiteboxFragment(_)
            | TileKind::RowWise(_)
            | TileKind::Bounce(_)
            | TileKind::Stage(_)
            | TileKind::Partition(_)
            | TileKind::Pipelined(_) => {
                panic!("PlaneVecTile::copy_from: unsupported source variant")
            }
        }
    }

    pub fn init_zero(&mut self) {
        planevec_load_zeros::<N>(&mut self.data, self.tile_size);
    }
}

// ===========================================================================
// Compute: matmul / load / write / zero-init
// ===========================================================================

#[cube]
pub fn planevec_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NPlaneVec>>,
    rhs: &Array<Vector<R, NPlaneVec>>,
    acc: &mut Array<Vector<A, NPlaneVec>>,
    #[comptime] tile_size: TileSize,
) {
    let n = tile_size.n();
    #[unroll]
    for n_idx in 0..n as usize {
        let mut acc_vec = acc[n_idx];
        #[unroll]
        for vi in 0..NPlaneVec::value() {
            let lhs_elem = A::cast_from(lhs[0usize].extract(vi));
            let rhs_elem = A::cast_from(rhs[n_idx].extract(vi));
            acc_vec.insert(vi, acc_vec.extract(vi) + plane_sum(lhs_elem * rhs_elem));
        }
        acc[n_idx] = acc_vec;
    }
}

#[cube]
pub fn planevec_load_from_shared<E: Numeric, ES: Size, N: Numeric>(
    shared: &SharedTile<E>,
    arr: &mut Array<Vector<N, NPlaneVec>>,
    #[comptime] tile_size: TileSize,
    #[comptime] ident: StageIdent,
) {
    let shared = shared.view::<ES>();
    let shared = &shared;
    match ident {
        StageIdent::Lhs => {
            let offset = shared.stage_offset(UNIT_POS_X);
            arr[0usize] = Vector::cast_from(shared.container[offset as usize]);
        }
        StageIdent::Rhs | StageIdent::Acc => {
            let n = tile_size.n();
            #[unroll]
            for n_idx in 0..n {
                let offset = shared.stage_offset(UNIT_POS_X + n_idx * shared.stride);
                arr[n_idx as usize] = Vector::cast_from(shared.container[offset as usize]);
            }
        }
        _ => panic!("Invalid ident for PlaneVec load"),
    }
}

#[cube]
pub fn planevec_load_zeros<N: Numeric>(
    arr: &mut Array<Vector<N, NPlaneVec>>,
    #[comptime] tile_size: TileSize,
) {
    let n = tile_size.n();
    let zero = N::from_int(0);
    #[unroll]
    for n_idx in 0..n as usize {
        arr[n_idx] = Vector::cast_from(zero);
    }
}

#[cube]
pub fn planevec_write_to_shared<A: Numeric, E: Numeric, ES: Size>(
    shared: &mut SharedTile<E>,
    arr: &Array<Vector<A, NPlaneVec>>,
    #[comptime] tile_size: TileSize,
    #[comptime] reduce_vector_size: u32,
) {
    let mut shared = shared.view::<ES>();
    let shared = &mut shared;
    if UNIT_POS_X == 0 {
        let out_vector_size = shared.container.vector_size().comptime();
        let n = tile_size.n();
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
                    sum += acc_vec.extract(i);
                }
                out_vector.insert(within_vector, E::cast_from(sum));
            }
            let offset = shared.stage_offset(out_vector_iter as u32);
            shared.container[offset as usize] = out_vector;
        }
    }
}
