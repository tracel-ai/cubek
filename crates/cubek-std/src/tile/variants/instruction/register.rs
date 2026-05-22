use cubecl::prelude::*;

use crate::{
    MatrixLayout, StageIdent, TileSize,
    tile::{
        Plane, RowWise, SharedTile, StridedTile, Tile, TileKind, TileKindExpand, TileScope,
        mask::Mask,
        variants::unit::{UnitTile, UnitTileLayout},
    },
};

/// Execution mode for the register-resident matmul. Lives at tile level
/// because the load + execute paths branch on it directly.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum ProductType {
    /// Computes the Tile Matmul as m*n inner products of length k.
    ///
    /// Needs Lhs to be row major and Rhs to be col major
    /// If not the case, tile will be transposed during load
    Inner,
    /// Computes the Stage Matmul as the sum of k outer products of size m*n.
    ///
    /// Needs Lhs to be col major and Rhs to be row major
    /// If not the case, tile will be transposed during load
    Outer,
}

impl ProductType {
    pub fn from_layouts(
        lhs_layout: MatrixLayout,
        rhs_layout: MatrixLayout,
        tile_size: TileSize,
    ) -> Self {
        let lhs_preferred = match lhs_layout {
            MatrixLayout::RowMajor => ProductType::Inner,
            MatrixLayout::ColMajor => ProductType::Outer,
        };
        let rhs_preferred = match rhs_layout {
            MatrixLayout::RowMajor => ProductType::Outer,
            MatrixLayout::ColMajor => ProductType::Inner,
        };

        if lhs_preferred == rhs_preferred {
            lhs_preferred
        } else if tile_size.m() == 1 {
            rhs_preferred
        } else if tile_size.n() == 1 {
            lhs_preferred
        } else {
            // No better solution
            ProductType::Outer
        }
    }
}

/// Register-resident matmul tile. Built on top of [`UnitTile`] (per-unit full
/// register copy of the data). Holds only the minimal comptime data the tile
/// body uses (`tile_size` + `product_type`); rowwise / elementwise ops are
/// delegated to the inner [`UnitTile`]. The matmul-level configuration lives
/// in cubek-matmul as `RegisterMatmul`.
#[derive(CubeType)]
pub struct RegisterTile<N: Numeric> {
    pub tile: UnitTile<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub tile_size: TileSize,
    #[cube(comptime)]
    pub product_type: ProductType,
}

#[cube]
impl<E: Float> RegisterTile<E> {
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        self.tile.row_max(acc, base);
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        self.tile.row_sum(acc);
    }

    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        self.tile.exp_diff(rowwise);
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        self.tile.rowwise_scale(scale);
    }

    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        self.tile.scale_and_mask::<M>(scale, mask);
    }

    pub fn fill_zero(&mut self) {
        self.tile.fill_zero();
    }

    /// Cast-copies this register tile into `dest`. Used by per-variant softmax
    /// helpers when writing the post-softmax score into a same-storage
    /// destination.
    pub fn write_to<Lhs: Float>(&self, dest: &mut RegisterTile<Lhs>) {
        self.tile.write_to::<Lhs>(&mut dest.tile);
    }
}

#[cube]
impl<A: Numeric> RegisterTile<A> {
    /// Executes `lhs · rhs`, accumulating into `self` via the configured
    /// inner/outer software product.
    pub fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &RegisterTile<L>, rhs: &RegisterTile<R>) {
        register_execute(
            &lhs.tile.data,
            &rhs.tile.data,
            &mut self.tile.data,
            self.tile_size,
            self.product_type,
        );
    }
}

#[cube]
impl<N: Numeric> RegisterTile<N> {
    /// Copies into the register tile from `source`. Supported sources:
    /// `SharedMemory` (product-type-aware load) and `None` (zero-init).
    pub fn copy_from<SE: Numeric, SS: Size, Sc: TileScope>(
        &mut self,
        source: &Tile<SE, Sc>,
        #[comptime] ident: StageIdent,
    ) {
        match &source.kind {
            TileKind::SharedTile(shared) => {
                register_load_from_shared::<SE, SS, N>(
                    shared,
                    &mut self.tile.data,
                    self.matrix_layout,
                    self.tile_size,
                    self.product_type,
                    ident,
                );
            }
            TileKind::None => {
                register_load_zeros::<N>(&mut self.tile.data, self.tile_size, ident);
            }
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
                panic!("RegisterTile::copy_from: unsupported source variant")
            }
        }
    }

    pub fn init_zero(&mut self, #[comptime] ident: StageIdent) {
        register_load_zeros::<N>(&mut self.tile.data, self.tile_size, ident);
    }
}

#[cube]
impl<Acc: Float> RegisterTile<Acc> {
    /// Online softmax for the Register variant (legacy direct-register
    /// attention path). Destination must be another `RegisterTile`.
    pub fn softmax<Lhs: Float, M: Mask>(
        &mut self,
        mask: &M,
        softmaxed: &mut Tile<Lhs, Plane>,
        state: &mut (RowWise<Acc>, RowWise<Acc>),
        head_dim_factor: Acc,
    ) -> RowWise<Acc> {
        let num_rows = comptime!(state.0.num_rows);
        let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
        let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

        self.scale_and_mask::<M>(head_dim_factor, mask);
        self.row_max(&mut max_buf, &state.0);
        self.exp_diff(&max_buf);
        self.row_sum(&mut sum_buf);

        let exp_m_diff = state.0.exp_diff(&max_buf);
        let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

        match &mut softmaxed.kind {
            TileKind::Register(d) => self.write_to::<Lhs>(d),
            TileKind::Bounce(_) => {
                panic!("RegisterTile::softmax: Bounce destination not supported")
            }
            TileKind::WhiteboxFragment(_) => {
                panic!("RegisterTile::softmax: WhiteboxFragment destination not supported")
            }
            TileKind::Unit(_) => panic!("RegisterTile::softmax: Unit destination not supported"),
            _ => panic!("RegisterTile::softmax: unsupported softmaxed variant"),
        }

        RowWise::copy_from(&mut state.0, &max_buf);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }
}

#[cube]
pub fn register_allocate_lhs<L: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] product_type: ProductType,
) -> Tile<L, Sc> {
    let m = comptime!(tile_size.m());
    let k = comptime!(tile_size.k());
    let inner_layout = comptime!(UnitTileLayout::new(m, k, false));
    Tile::from_kind(TileKind::new_Register(RegisterTile::<L> {
        tile: UnitTile::<L>::new(inner_layout),
        matrix_layout: layout,
        tile_size,
        product_type,
    }))
}

#[cube]
pub fn register_allocate_rhs<R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] product_type: ProductType,
) -> Tile<R, Sc> {
    let n = comptime!(tile_size.n());
    let k = comptime!(tile_size.k());
    let inner_layout = comptime!(UnitTileLayout::new(n, k, false));
    Tile::from_kind(TileKind::new_Register(RegisterTile::<R> {
        tile: UnitTile::<R>::new(inner_layout),
        matrix_layout: layout,
        tile_size,
        product_type,
    }))
}

#[cube]
pub fn register_allocate_acc<A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] product_type: ProductType,
) -> Tile<A, Sc> {
    let m = comptime!(tile_size.m());
    let n = comptime!(tile_size.n());
    let inner_layout = comptime!(UnitTileLayout::new(m, n, false));
    Tile::from_kind(TileKind::new_Register(RegisterTile::<A> {
        tile: UnitTile::<A>::new(inner_layout),
        matrix_layout: layout,
        tile_size,
        product_type,
    }))
}

// ===========================================================================
// Compute: matmul / load / write / zero-init
// ===========================================================================

pub(crate) const UNROLL: bool = false;

#[cube]
pub fn register_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    rhs: &Array<R>,
    acc: &mut Array<A>,
    #[comptime] tile_size: TileSize,
    #[comptime] product_type: ProductType,
) {
    let m = tile_size.m();
    let n = tile_size.n();
    let k = tile_size.k();
    match product_type {
        ProductType::Inner => {
            inner_product::<L, R, A>(lhs, rhs, acc, m, n, k);
        }
        ProductType::Outer => {
            outer_product::<L, R, A>(lhs, rhs, acc, m, n, k);
        }
    }
}

#[cube]
fn inner_product<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    rhs: &Array<R>,
    acc: &mut Array<A>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    #[unroll(UNROLL)]
    for m_ in 0..m as usize {
        #[unroll(UNROLL)]
        for n_ in 0..n as usize {
            #[unroll(UNROLL)]
            for k_ in 0..k as usize {
                let lhs_elem = A::cast_from(lhs[m_ * k as usize + k_]);
                let rhs_elem = A::cast_from(rhs[n_ * k as usize + k_]);
                acc[m_ * n as usize + n_] += lhs_elem * rhs_elem;
            }
        }
    }
}

#[cube]
fn outer_product<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<L>,
    rhs: &Array<R>,
    acc: &mut Array<A>,
    #[comptime] m: u32,
    #[comptime] n: u32,
    #[comptime] k: u32,
) {
    #[unroll(UNROLL)]
    for k_ in 0..k as usize {
        #[unroll(UNROLL)]
        for m_ in 0..m as usize {
            let lhs_elem = A::cast_from(lhs[k_ * m as usize + m_]);
            #[unroll(UNROLL)]
            for n_ in 0..n as usize {
                let rhs_elem = A::cast_from(rhs[k_ * n as usize + n_]);
                acc[m_ * n as usize + n_] += lhs_elem * rhs_elem;
            }
        }
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
pub fn register_load_from_shared<E: Numeric, ES: Size, N: Numeric>(
    shared: &SharedTile<E>,
    arr: &mut Array<N>,
    #[comptime] matrix_layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
    #[comptime] product_type: ProductType,
    #[comptime] ident: StageIdent,
) {
    let shared = shared.view::<ES>();
    let shared = &shared;
    let m = tile_size.m();
    let n = tile_size.n();
    let k = tile_size.k();

    match ident {
        StageIdent::Lhs => match product_type {
            ProductType::Inner => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_plain::<E, ES, N>(shared, arr, m, k);
                }
                MatrixLayout::ColMajor => {
                    load_transposed::<E, ES, N>(shared, arr, k, m);
                }
            },
            ProductType::Outer => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_transposed::<E, ES, N>(shared, arr, m, k);
                }
                MatrixLayout::ColMajor => {
                    load_plain::<E, ES, N>(shared, arr, k, m);
                }
            },
        },
        StageIdent::Rhs => match product_type {
            ProductType::Inner => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_transposed::<E, ES, N>(shared, arr, k, n);
                }
                MatrixLayout::ColMajor => {
                    load_plain::<E, ES, N>(shared, arr, n, k);
                }
            },
            ProductType::Outer => match matrix_layout {
                MatrixLayout::RowMajor => {
                    load_plain::<E, ES, N>(shared, arr, k, n);
                }
                MatrixLayout::ColMajor => {
                    load_transposed::<E, ES, N>(shared, arr, n, k);
                }
            },
        },
        StageIdent::Acc => match matrix_layout {
            MatrixLayout::RowMajor => {
                load_plain::<E, ES, N>(shared, arr, m, n);
            }
            MatrixLayout::ColMajor => {
                load_transposed::<E, ES, N>(shared, arr, n, m);
            }
        },
        _ => panic!("Invalid ident for Register load"),
    }
}

#[cube]
fn load_plain<E: Numeric, ES: Size, N: Numeric>(
    tile: &StridedTile<E, ES>,
    arr: &mut Array<N>,
    #[comptime] num_segments: u32,
    #[comptime] segment_size: u32,
) {
    let line_size = ES::value() as u32;
    let num_lines_per_segment = segment_size / line_size;

    #[unroll(UNROLL)]
    for segment in 0..num_segments {
        #[unroll(UNROLL)]
        for line_within_segment in 0..num_lines_per_segment {
            let line = tile.get_vector(segment, line_within_segment);
            #[unroll]
            for pos_within_line in 0..line_size {
                arr[(segment * segment_size + line_within_segment * line_size + pos_within_line)
                    as usize] = N::cast_from(line.extract(pos_within_line as usize));
            }
        }
    }
}

#[cube]
fn load_transposed<E: Numeric, ES: Size, N: Numeric>(
    tile: &StridedTile<E, ES>,
    arr: &mut Array<N>,
    #[comptime] num_segments: u32,
    #[comptime] segment_size: u32,
) {
    let line_size = ES::value() as u32;
    let num_lines_per_segment = segment_size / line_size;

    #[unroll(UNROLL)]
    for segment in 0..num_segments {
        #[unroll(UNROLL)]
        for line_within_segment in 0..num_lines_per_segment {
            let line = tile.get_vector(segment, line_within_segment);
            #[unroll]
            for pos_within_line in 0..line_size {
                arr[((line_within_segment * line_size + pos_within_line) * num_segments + segment)
                    as usize] = N::cast_from(line.extract(pos_within_line as usize));
            }
        }
    }
}

#[cube]
pub fn register_load_zeros<N: Numeric>(
    arr: &mut Array<N>,
    #[comptime] tile_size: TileSize,
    #[comptime] ident: StageIdent,
) {
    let size = match ident {
        StageIdent::Lhs => tile_size.m() * tile_size.k(),
        StageIdent::Rhs => tile_size.n() * tile_size.k(),
        StageIdent::Acc | StageIdent::Out => tile_size.m() * tile_size.n(),
    };
    for i in 0..size {
        arr[i as usize] = N::from_int(0);
    }
}

#[cube]
pub fn register_write_to_shared<E: Numeric, ES: Size, A: Numeric>(
    shared: &mut SharedTile<E>,
    arr: &Array<A>,
    #[comptime] tile_size: TileSize,
) {
    let mut shared = shared.view::<ES>();
    let shared = &mut shared;
    let out_vector_size = shared.container.vector_size().comptime() as u32;
    let size_mn = tile_size.m() * tile_size.n();

    #[unroll(false)]
    for i in 0..size_mn / out_vector_size {
        let offs = shared.stage_offset(i);
        let mut vector = Vector::<A, ES>::empty();
        #[unroll]
        for j in 0..out_vector_size {
            vector.insert(j as usize, arr[(i * out_vector_size + j) as usize]);
        }
        shared.container[offs as usize] = Vector::cast_from(vector);
    }
}
