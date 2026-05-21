use cubecl;
use cubecl::{prelude::*, std::tensor::layout::Coords2d};

use crate::tile::LOGIT_MASKED;
use crate::tile::{
    Plane, RowWise, StridedTile, Tile, TileKind, TileKindExpand,
    mask::{Mask, MaskExpand},
    scope::{TileScope, assert_plane_scope},
};

#[derive(CubeType)]
/// Assumes:
/// - unit_size * plane_dim = total_size (not dim wise but in total count)
pub struct WhiteboxFragment<E: Numeric> {
    pub array: Array<E>,
    #[cube(comptime)]
    pub layout: WhiteboxFragmentLayout,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum InnerLayout {
    /// Each unit has all its elements contiguous inside the same row
    ///
    ///  0,  0,  1,  1,  2,  2,  3,  3,
    ///  4,  4,  5,  5,  6,  6,  7,  7,
    ///  8,  8,  9,  9, 10, 10, 11, 11,
    /// 12, 12, 13, 13, 14, 14, 15, 15,
    /// 16, 16, 17, 17, 18, 18, 19, 19,
    /// 20, 20, 21, 21, 22, 22, 23, 23,
    /// 24, 24, 25, 25, 26, 26, 27, 27,
    /// 28, 28, 29, 29, 30, 30, 31, 31,
    Contiguous,
    /// Each unit spreads its elements along two rows
    ///
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    ///  0,  1,  2,  3,  4,  5,  6,  7,
    ///  8,  9, 10, 11, 12, 13, 14, 15,
    /// 16, 17, 18, 19, 20, 21, 22, 23,
    /// 24, 25, 26, 27, 28, 29, 30, 31,
    SplitRows,
}

#[cube]
impl<E: Numeric> WhiteboxFragment<E> {
    pub fn new(#[comptime] layout: WhiteboxFragmentLayout) -> WhiteboxFragment<E> {
        let array = Array::<E>::new(comptime!(layout.unit_size.0 * layout.unit_size.1) as usize);

        WhiteboxFragment::<E> { array, layout }
    }

    pub fn zero(&mut self) {
        for i in 0..self.layout.unit_size.0 * self.layout.unit_size.1 {
            self.array[i as usize] = E::from_int(0);
        }
    }

    pub fn load_from_slice(&mut self, smem_slice: &[E]) {
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = whitebox_fragment_absolute_pos(self.layout, (r, c));
                let index = row * self.layout.total_size.1 + col;

                self.array[(r * self.layout.unit_size.1 + c) as usize] = smem_slice[index as usize];
            }
        }
    }

    pub fn load_from_strided_tile<E2: Numeric, N: Size>(
        &mut self,
        strided_tile: &StridedTile<E2, N>,
    ) {
        // Assumes vector size == 1
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = whitebox_fragment_absolute_pos(self.layout, (r, c));
                self.array[(r * self.layout.unit_size.1 + c) as usize] =
                    E::cast_from(strided_tile.get_vector(row, col))
            }
        }
    }

    pub fn store_to<F: Float>(&self, smem_slice: &mut [F]) {
        for r in 0..self.layout.unit_size.0 {
            for c in 0..self.layout.unit_size.1 {
                let (row, col) = whitebox_fragment_absolute_pos(self.layout, (r, c));
                let index = row * self.layout.total_size.1 + col;

                smem_slice[index as usize] =
                    F::cast_from(self.array[(r * self.layout.unit_size.1 + c) as usize]);
            }
        }
    }

    /// Reads the element at `local_pos` and casts to `bool`. Used by the
    /// `Mask` trait dispatcher when this fragment is acting as a
    /// materialized mask fragment.
    pub fn should_mask(&self, local_pos: Coords2d) -> bool {
        bool::cast_from(self.array[(local_pos.0 * self.layout.unit_size.1 + local_pos.1) as usize])
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        for r in 0..self.layout.unit_size.0 as usize {
            let row_offset = r as u32 * self.layout.unit_size.1;
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index as usize] *= scale.vals[r];
            }
        }
    }

    pub fn rowwise_max(&self) -> RowWise<E> {
        let num_rows = comptime!(self.layout.unit_size.0) as usize;
        let num_cols = comptime!(self.layout.unit_size.1) as usize;
        let mut vals = Array::<E>::new(num_rows);

        for r in 0..num_rows {
            let row_offset = r * num_cols;
            let mut val = E::min_value();

            for c in 0..num_cols {
                let index = row_offset + c;
                val = max(val, self.array[index]);
            }

            vals[r] = val;
        }

        RowWise::<E> { num_rows, vals }
    }

    pub fn rowwise_sum(&self) -> RowWise<E> {
        let num_rows = comptime!(self.layout.unit_size.0) as usize;
        let num_cols = comptime!(self.layout.unit_size.1) as usize;
        let mut vals = Array::<E>::new(num_rows);

        for r in 0..num_rows {
            let row_offset = r * num_cols;
            let mut val = E::from_int(0);

            for c in 0..num_cols {
                let index = row_offset + c;
                val += self.array[index];
            }

            vals[r] = val;
        }

        RowWise::<E> { num_rows, vals }
    }

    pub fn num_units_per_row(&self) -> comptime_type!(u32) {
        comptime!(self.layout.total_size.1 / self.layout.unit_size.1)
    }

    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        for r in 0..self.layout.unit_size.0 {
            let row_offset = r * self.layout.unit_size.1;
            for c in 0..self.layout.unit_size.1 {
                let index = row_offset + c;
                self.array[index as usize] = self.array[index as usize] * scale
                    + E::cast_from(mask.should_mask((r, c))) * E::min_value();
            }
        }
    }
}

#[cube]
impl<E: Float> WhiteboxFragment<E> {
    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        let num_rows = comptime!(self.layout.unit_size.0) as usize;
        let num_cols = comptime!(self.layout.unit_size.1) as usize;
        let threshold = E::new(LOGIT_MASKED);

        for r in 0..num_rows {
            let row_offset = r * num_cols;

            let val = rowwise.vals[r];
            let safe_val = clamp_min(val, threshold);
            let not_masked = E::cast_from(val >= threshold);

            for c in 0..num_cols {
                let index = row_offset + c;

                self.array[index] = not_masked * (self.array[index] - safe_val).exp();
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct WhiteboxFragmentLayout {
    pub total_size: Coords2d,
    pub unit_size: Coords2d,
    pub num_units_per_row: u32,
    pub plane_dim: u32,
}

impl WhiteboxFragmentLayout {
    pub const fn new(
        total_size: Coords2d,
        plane_dim: u32,
        inner_layout: InnerLayout,
    ) -> WhiteboxFragmentLayout {
        let total_elements = total_size.0 * total_size.1;
        let elements_per_unit = total_elements.div_ceil(plane_dim);

        let (num_rows_per_unit, num_cols_per_unit) = match inner_layout {
            InnerLayout::Contiguous => (1u32, elements_per_unit),
            InnerLayout::SplitRows => (2u32, elements_per_unit / 2u32),
        };
        let unit_size = (num_rows_per_unit, num_cols_per_unit);
        let num_units_per_row = total_size.1 / unit_size.1;

        WhiteboxFragmentLayout {
            total_size,
            unit_size,
            num_units_per_row,
            plane_dim,
        }
    }

    pub const fn num_units_per_row(&self) -> u32 {
        self.total_size.1 / self.unit_size.1
    }
}

#[cube]
/// Allocates a `Tile::WhiteboxFragment` for the given scope. Panics at expansion
/// time unless `Sc = Plane`.
pub fn allocate_whitebox_fragment<E: Numeric, Sc: TileScope>(
    #[comptime] layout: WhiteboxFragmentLayout,
) -> Tile<E, Sc> {
    comptime!(assert_plane_scope(Sc::KIND));
    Tile::from_kind(TileKind::new_WhiteboxFragment(WhiteboxFragment::<E>::new(
        layout,
    )))
}

/// Maps a per-unit `(row, col)` to its absolute position within the tile
/// described by `layout`.
#[cube]
pub fn whitebox_fragment_absolute_pos(
    #[comptime] layout: WhiteboxFragmentLayout,
    local_pos: Coords2d,
) -> Coords2d {
    let abs_row_index = {
        let row_0 = UNIT_POS_X / layout.num_units_per_row;
        let row_jump = comptime!(layout.plane_dim / layout.num_units_per_row);
        local_pos.0 * row_jump + row_0
    };
    let abs_col_index = layout.unit_size.1 * (UNIT_POS_X % layout.num_units_per_row) + local_pos.1;
    (abs_row_index, abs_col_index)
}

/// Zeroes a slice giving responsibility to units following `layout`.
#[cube]
pub fn whitebox_fragment_zero_slice<E: Numeric>(
    #[comptime] layout: WhiteboxFragmentLayout,
    slice: &mut [E],
) {
    for r in 0..layout.unit_size.0 {
        for c in 0..layout.unit_size.1 {
            let (row, col) = whitebox_fragment_absolute_pos(layout, (r, c));
            let index = row * layout.total_size.1 + col;

            slice[index as usize] = E::from_int(0);
        }
    }
}

// ===========================================================================
// Cross-plane row reduction
//
// Reduces row-wise quantities across plane units that share a row, masking
// out off-row peers. Restricted to plane scope (uses `plane_shuffle` and
// `UNIT_POS_X`); callers enforce that.
// ===========================================================================

#[cube]
impl<E: Float> WhiteboxFragment<E> {
    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        acc.copy_from(base);
        reduce::<E, FragmentRowMax>(acc, self);
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        acc.fill(E::from_int(0));
        reduce::<E, FragmentRowSum>(acc, self);
    }
}

#[cube]
fn reduce<E: Float, RO: ReduceOp<E>>(vals: &mut RowWise<E>, data: &WhiteboxFragment<E>) {
    let num_units_per_row = data.num_units_per_row().comptime();
    let num_shares_within_plane = num_units_per_row.next_power_of_two().ilog2();

    let unit_pos = UNIT_POS_X;
    let unit_pos_in_row = unit_pos % num_units_per_row;

    RO::reduce_local(data, vals);

    for i in 0..num_shares_within_plane {
        let offset = num_units_per_row >> (i + 1);
        let source_unit = unit_pos + offset;

        let value_from_source = rowwise_plane_broadcast(&*vals, source_unit);

        // Mask if outside the row
        let mask = unit_pos_in_row + offset >= num_units_per_row;
        RO::reduce_from_peer(vals, &value_from_source, mask);
    }

    // Broadcast back to subgroup
    let result = rowwise_plane_broadcast(&*vals, unit_pos - unit_pos_in_row);
    vals.copy_from(&result);
}

#[cube]
fn rowwise_plane_broadcast<E: Float>(rowwise: &RowWise<E>, source_unit: u32) -> RowWise<E> {
    let mut result = Array::<E>::new(rowwise.num_rows);

    for r in 0..rowwise.num_rows {
        result[r] = plane_shuffle(rowwise.vals[r], source_unit);
    }

    RowWise::<E> {
        num_rows: rowwise.num_rows,
        vals: result,
    }
}

#[cube]
trait ReduceOp<E: Float> {
    fn reduce_local(data: &WhiteboxFragment<E>, acc: &mut RowWise<E>);
    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool);
}

#[derive(CubeType)]
struct FragmentRowMax {}

#[derive(CubeType)]
struct FragmentRowSum {}

#[cube]
impl<E: Float> ReduceOp<E> for FragmentRowMax {
    fn reduce_local(data: &WhiteboxFragment<E>, acc: &mut RowWise<E>) {
        acc.max_inplace(&data.rowwise_max())
    }

    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(mask) * E::min_value());
        masked.add_inplace(elem);

        acc.max_inplace(&masked)
    }
}

#[cube]
impl<E: Float> ReduceOp<E> for FragmentRowSum {
    fn reduce_local(data: &WhiteboxFragment<E>, acc: &mut RowWise<E>) {
        acc.add_inplace(&data.rowwise_sum())
    }

    fn reduce_from_peer(acc: &mut RowWise<E>, elem: &RowWise<E>, mask: bool) {
        let mut masked = RowWise::new_filled(elem.num_rows, E::cast_from(!mask));
        masked.mul_inplace(elem);

        acc.add_inplace(&masked)
    }
}

// ===========================================================================
// Online softmax over a free-standing WhiteboxFragment score.
// ===========================================================================

#[cube]
impl<Acc: Float> WhiteboxFragment<Acc> {
    /// Online softmax for a free-standing WhiteboxFragment score (the
    /// register-only variant of attention's softmax). Writes the post-softmax
    /// values into `softmaxed` (which may be `Bounce` — routed through its
    /// smem into its cmma fragment — or another `WhiteboxFragment`).
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

        write_fragment_into::<Acc, Lhs>(&*self, softmaxed);

        RowWise::copy_from(&mut state.0, &max_buf);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }
}

/// Write post-softmax `WhiteboxFragment` values into `softmaxed`.
#[cube]
fn write_fragment_into<Acc: Float, Lhs: Float>(
    src: &WhiteboxFragment<Acc>,
    softmaxed: &mut Tile<Lhs, Plane>,
) {
    match &mut softmaxed.kind {
        TileKind::Bounce(d) => {
            let stride = comptime!(d.cmma.tile_size.n());
            src.store_to(&mut d.smem);
            sync_cube();
            cubecl::cmma::load(&mut d.cmma.matrix, &d.smem, stride);
        }
        TileKind::WhiteboxFragment(d) => {
            let total = comptime!(src.layout.unit_size.0 * src.layout.unit_size.1);
            for i in 0..total {
                d.array[i as usize] = Lhs::cast_from(src.array[i as usize]);
            }
        }
        _ => panic!("write_fragment_into: unsupported softmaxed variant"),
    }
}
