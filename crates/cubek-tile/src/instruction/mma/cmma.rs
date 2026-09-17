//! The tensor-core leaf: `acc += lhs · rhs` via `cmma::execute`. The accumulator is
//! always a resident fragment; the operands arrive as fragments or as staged smem
//! windows (row-major by construction), the latter loaded into transient `A`/`B`
//! fragments here. A gmem window's layout is unchecked, so it must be staged first.
//!
//! The rhs window is `{k, n}` as stored, or `{n, k}` when its trailing axis is the one
//! contracted: that window is read as a col-major `B`, so `a · bᵀ` needs no transposed
//! copy ([`rhs_layout`]).

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
    std::tensor::layout::CoordsDyn,
};

use crate::instruction::registers::contract::{
    Side, check_scales_omit_rather_than_divide, check_scales_ride, scale_width,
};
use crate::*;

#[cube]
impl<A: Numeric> CmmaData<A> {
    /// Tensor-core contraction `self += lhs · rhs`, each factor times whatever scales it carries.
    ///
    /// Operands already resident as fragments execute as they are. Every other operand is read
    /// into a transient `A`/`B` fragment first, which is where a scaled factor is served: a
    /// fragment load reads memory as it lies, so a factor carrying scales is folded into its
    /// plane's landing and the load reads that. Marlin's shape, with the landing where Metal's
    /// fragment load wants memory.
    pub(crate) fn mma<EL: Numeric, LS: Numeric, ER: Numeric, RS: Numeric>(
        &self,
        lhs: &Scaled<EL, LS>,
        rhs: &Scaled<ER, RS>,
        #[comptime] out: Space,
    ) {
        let lhs_values = lhs.values();
        let rhs_values = rhs.values();
        match (&lhs_values.tile_kind, &rhs_values.tile_kind) {
            (TileKind::PlaneTile(a), TileKind::PlaneTile(b)) => match (a, b) {
                (PlaneTile::Cmma(a), PlaneTile::Cmma(b)) => {
                    lhs.refuse_scales();
                    rhs.refuse_scales();
                    cmma::execute(&a.matrix, &b.matrix, &self.matrix, &self.matrix)
                }
                _ => panic!("cmma operands must be cmma fragments"),
            },
            _ => {
                let a_read = comptime!(FragmentRead::new(
                    Side::Lhs,
                    &lhs_values.space,
                    &rhs_values.space,
                    &out
                ));
                let b_read = comptime!(FragmentRead::new(
                    Side::Rhs,
                    &lhs_values.space,
                    &rhs_values.space,
                    &out
                ));
                let mut a_frag = unsafe {
                    Matrix::<EL>::uninitialized(
                        MatrixIdent::A,
                        a_read.m,
                        a_read.n,
                        a_read.k,
                        MatrixLayout::RowMajor,
                    )
                };
                let mut b_frag = unsafe {
                    Matrix::<ER>::uninitialized(
                        MatrixIdent::B,
                        b_read.m,
                        b_read.n,
                        b_read.k,
                        b_read.layout,
                    )
                };
                lhs.load(&mut a_frag, comptime!(a_read));
                rhs.load(&mut b_frag, comptime!(b_read));
                cmma::execute(&a_frag, &b_frag, &self.matrix, &self.matrix);
            }
        }
    }
}

/// How a rhs window is read: col-major when its trailing axis is `contracted`, the matrix
/// then being the window's transpose; row-major otherwise. The rule
/// [`PlanePartition::store`](crate::PlanePartition::store) loads a `B` fragment by.
pub(crate) fn rhs_layout(rhs: &Space, contracted: Axis) -> MatrixLayout {
    match rhs.axis_at(rhs.rank() - 1) == contracted {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    }
}

/// How one factor of a tensor-core contraction is read into its fragment: the fragment's own
/// edges, the matrix this factor's window is, and what the scales it carries are checked
/// against.
///
/// Read off the two operand spaces and the accumulator's, once per factor, so neither can
/// disagree with the other about the contraction they share. `side` is stated by the caller —
/// it is the factor it is calling on — and nothing here infers it.
#[derive(Clone)]
pub(crate) struct FragmentRead {
    /// The fragment's edges, the same for both factors.
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// The layout the rhs window reads at; the lhs is always row-major.
    pub layout: MatrixLayout,
    /// This factor's own window as a matrix: the lhs is `m × k`; the rhs is `k × n`, or `n × k`
    /// where the contraction is its trailing axis and the fragment reads it col-major.
    pub rows: usize,
    pub cols: usize,
    /// What the scales riding this factor are checked against: the side they were stated on,
    /// and the accumulator's matrix.
    pub side: Side,
    pub out: Space,
    pub acc_axes: MatrixAxes,
}

impl FragmentRead {
    pub(crate) fn new(side: Side, lhs: &Space, rhs: &Space, out: &Space) -> Self {
        // The fragment's edges off the accumulator's axes and the contracted extent, as the plain
        // leaf reads them: a split contraction is one `k` edge, whatever its digits.
        let acc_axes = MatrixAxes::accumulator(out, lhs);
        let m = acc_axes.rows(out);
        let n = acc_axes.cols(out);
        let operands = Space::merge(&[lhs, rhs]);
        let k = operands.contracted_extent(out);
        let layout = rhs_layout(rhs, lhs.axis_at(lhs.rank() - 1));
        let transposed = layout == MatrixLayout::ColMajor;
        // The lhs window is `m × k` in the order it lies; the rhs is `k × n`, or its transpose
        // where the contraction is its trailing axis.
        let (rows, cols) = match (side, transposed) {
            (Side::Lhs, _) => (m, k),
            (Side::Rhs, true) => (n, k),
            (Side::Rhs, false) => (k, n),
        };
        FragmentRead {
            m,
            n,
            k,
            layout,
            rows,
            cols,
            side,
            out: out.clone(),
            acc_axes,
        }
    }
}

#[cube]
impl<E: Numeric, S: Numeric> Scaled<E, S> {
    /// This factor read into `frag`.
    ///
    /// A fragment loads a window as it lies, so the window's layout has to be one the
    /// instruction can be told: a shared one is, and a global one is not. **The landing is how a
    /// factor answers that** ([`land`](Scaled::land)) — it holds the factor's own values in the
    /// plane's shared window, row-major by construction, and the load reads them from there.
    /// A factor carrying scales has no other route, since its values do not exist anywhere
    /// until they are scaled; one carrying none takes it when it has a landing, and is read
    /// from its window as it lies when the window is already shared.
    pub(crate) fn load(&self, frag: &mut Matrix<E>, #[comptime] read: FragmentRead) {
        let values = self.values();
        let count = self.levels().len();
        let landed = values.has_landing();
        let packing = values.packing();
        if comptime!(count > 0 || landed) {
            let landing = self.land(comptime!(read.clone()));
            cmma::load(frag, &landing, comptime!(read.cols as u32));
            // The landing is this region's until every lane's load has read it.
            sync_plane();
        } else {
            match &values.tile_kind {
                TileKind::Smem(m) => {
                    // A packed stage holds words, and a fragment loads a window as it lies: it
                    // lands first, or it is not read at all.
                    comptime!(assert!(
                        packing == Packing::Plain,
                        "mma: a packed stage reaches a fragment through a landing; open the \
                         operand with `with_landing`"
                    ));
                    cmma::load(frag, m.window_slice(), m.row_stride())
                }
                TileKind::Gmem(_) => panic!(
                    "mma: a fragment loads a window as it lies and a gmem layout is unchecked; \
                     open the operand with `with_landing()`, or stage it"
                ),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_) => {
                    panic!("mma: an operand reaches a fragment from a memory window")
                }
            }
        }
    }

    /// Refuse the scales this factor carries: a reader with no memory to fold them into cannot
    /// serve one.
    pub(crate) fn refuse_scales(&self) {
        let count = self.levels().len();
        comptime!(assert!(
            count == 0,
            "mma: an operand already resident as a fragment has no memory for a scale to fold \
             into; scale it where it was staged"
        ));
    }

    /// This factor in its plane's landing, row-major: each lane reads lines through the values'
    /// packed view, multiplies each by the scale covering it where the factor carries one, and
    /// writes them; the plane then syncs past the writes.
    ///
    /// A factor with no scales lands its values as they lie — the same walk, with nothing to
    /// multiply by — which is what makes the landing an operand's *residence* rather than a
    /// scale mechanism: it is the answer to a layout a fragment cannot be told, and a packed or
    /// scaled factor needs it for its values as well.
    ///
    /// **A scale is looked up at the coordinates of the value it covers.** The walk is over the
    /// landing's lines, in the order they lie, so neighbouring lanes write neighbouring words;
    /// each line's position in the values' space is what its scale is read at
    /// ([`ScaleLookup`]). No matrix of the scales is oriented anywhere: which of their axes is
    /// the block and which the column is answered by the coordinates, whatever order the two
    /// tensors state their axes in.
    fn land(&self, #[comptime] read: FragmentRead) -> Shared<[E]> {
        let values = self.values();
        let mut landing = match &values.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.landing(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("mma: an operand lands from its memory window")
            }
        };
        let levels = self.levels();

        let vw = values.vector_size();
        let size!(VW) = vw;
        let sw = scale_width(&levels);
        let size!(SW) = sw;
        let axes = comptime!(MatrixAxes::of(&values.space, read.rows, read.cols));
        let matrix = values.matrix_packed::<VW>(axes, 0usize);
        let scales = ScaleLookup::<S, SW>::of(
            &levels,
            comptime!(values.space.clone()),
            vw,
            comptime!(read.side),
            comptime!(read.out.clone()),
            comptime!(read.acc_axes),
        );

        let per_row = comptime!(read.cols / vw);
        let width = comptime!(vw as u32);
        let row_cells = comptime!(read.cols as u32);
        for cell in range_stepped(
            UNIT_POS_PLANE,
            comptime!((read.rows * per_row) as u32),
            PLANE_DIM,
        ) {
            let r = cell.fdiv(comptime!(per_row as u32));
            let c = cell.frem(comptime!(per_row as u32));
            let coords = value_coords(r, c, comptime!(values.space.clone()), axes, vw);
            let landed = scales.apply::<E, VW>(matrix.read((r, c)), &coords);
            let base = (r * row_cells + c * width) as usize;
            #[unroll]
            for j in 0..vw {
                landing[base + j] = landed.extract(j);
            }
        }
        sync_plane();
        landing
    }
}

/// The logical coordinate of the value line at `(row, col)` of a landing's matrix: one entry
/// per axis of the values' space, in scalars, the line's first value. The batch axes above the
/// matrix are pinned to the first, which is the one a landing holds.
#[cube]
fn value_coords(
    row: u32,
    col: u32,
    #[comptime] space: Space,
    #[comptime] axes: MatrixAxes,
    #[comptime] vw: usize,
) -> Coords<u32> {
    let rank = comptime!(space.rank());
    let mut coords = Coords::<u32>::new();
    #[unroll]
    for _batch in 0..comptime!(axes.row_split) {
        coords.push(0u32.runtime());
    }
    let rows = unravel_const(
        comptime!(
            (axes.row_split..axes.col_split)
                .map(|p| space.extent_at(p))
                .collect::<Vec<_>>()
        ),
        row,
    );
    #[unroll]
    for p in 0..rows.len() {
        coords.push(rows.at(p));
    }
    // The column edge counts in lines, so its innermost digit is a line index; the value's own
    // coordinate is that many lines in.
    let cols = unravel_const(
        comptime!(line_extents(&space, vw, axes.col_split, rank)),
        col,
    );
    let n = cols.len();
    #[unroll]
    for p in 0..n {
        if comptime!(p == n - 1) {
            coords.push(cols.at(p).fmul(comptime!(vw as u32)));
        } else {
            coords.push(cols.at(p));
        }
    }
    coords
}

/// The scales a landing multiplies by, read at the coordinates of the value they cover.
///
/// A scale is a tile in the values' space with the axes one scale holds whole omitted, so the
/// value's own coordinate names its scale: the coordinate of every axis the scales address is
/// the value's, and the axes they omit contribute nothing. The level nearest the values is read
/// as the lines its binding serves, and the scale is the field of that line the coordinate
/// falls in — a shift and a byte where the scales come four to a word. Every coarser level is
/// read once, at the region's origin, and carried as one value: a coarser level covers the
/// whole region, so it has no position of its own inside it.
///
/// Absent is a factor with no scales, and it emits nothing: its values go through as they lie.
#[derive(CubeType)]
struct ScaleLookup<'a, S: Numeric, W: Size> {
    /// The level nearest the values, where the factor carries any.
    inner: ComptimeOption<MaskedView<'a, Vector<S, W>, CoordsDyn>>,
    /// Every coarser level, already met and carried as one value.
    coarser: Vector<S, Const<1>>,
    /// The values' space, which every coordinate handed in is stated over.
    #[cube(comptime)]
    values: Space,
    /// The inner level's space, which the lookup is read over.
    #[cube(comptime)]
    scales: Option<Space>,
}

#[cube]
impl<'a, S: Numeric, W: Size> ScaleLookup<'a, S, W> {
    /// The lookup for a factor carrying `levels`, whose values lie in `values` and are served
    /// `vw` wide; `side`, `out` and `acc_axes` are what the scales' statement is checked against.
    fn of(
        levels: &'a Sequence<Tile<S>>,
        #[comptime] values: Space,
        #[comptime] vw: usize,
        #[comptime] side: Side,
        #[comptime] out: Space,
        #[comptime] acc_axes: MatrixAxes,
    ) -> Self {
        let count = levels.len();
        let mut coarser = Vector::<S, Const<1>>::cast_from(1);
        #[unroll]
        for k in 1..count {
            let above = levels.index(k);
            let rank = comptime!(above.space.rank());
            let above_width = above.vector_size();
            let size!(AW) = above_width;
            let mut origin = CoordsDyn::new();
            #[unroll]
            for _axis in 0..rank {
                origin.push(0u32.runtime());
            }
            // Read as the word it lies in, and its own scale is the word's first field.
            let one = above
                .nd_packed::<AW>(comptime!(Guard::Checked))
                .read(origin)
                .extract(0usize);
            coarser *= Vector::<S, Const<1>>::cast_from(one);
        }
        if comptime!(count > 0) {
            let inner = levels.index(0);
            let projection = inner.projection();
            comptime!(check_scales_omit_rather_than_divide(&projection));
            comptime!(check_scales_ride(side, &inner.space, &out, acc_axes));
            // A line of values is under one scale: the axis it runs along is one the scales
            // omit, or the line is one value.
            let innermost = comptime!(values.axis_at(values.rank() - 1));
            comptime!(assert!(
                vw == 1 || !projection.addresses(innermost),
                "mma: a landed line runs {vw} values along {innermost:?}, which its scales \
                 address, so one line lies under several scales; serve the factor one value a \
                 line, or omit {innermost:?} from the scales"
            ));
            comptime!(assert!(
                inner.space.axes().all(|axis| values.contains(axis)),
                "mma: the scales span {:?} where the values span {:?}; a scale is looked up at \
                 the value's coordinates, so every axis of the scales is one of the values'",
                inner.space.axes().collect::<Vec<_>>(),
                values.axes().collect::<Vec<_>>()
            ));
            ScaleLookup::<'a, S, W> {
                inner: ComptimeOption::new_Some(inner.nd_packed::<W>(comptime!(Guard::Checked))),
                coarser,
                values,
                scales: comptime!(Some(inner.space.clone())),
            }
        } else {
            ScaleLookup::<'a, S, W> {
                inner: ComptimeOption::new_None(),
                coarser,
                values,
                scales: comptime!(None),
            }
        }
    }

    /// `value`, the line whose first value lies at `coords`, under the scale covering it.
    fn apply<E: Numeric, V: Size>(
        &self,
        value: Vector<E, V>,
        coords: &Coords<u32>,
    ) -> Vector<E, V> {
        #[comptime]
        match &self.inner {
            ComptimeOption::Some(view) => {
                let sw = W::value();
                let (pos, field) = scale_coords(
                    coords,
                    comptime!(self.values.clone()),
                    comptime!(self.scales.clone().unwrap()),
                    sw,
                );
                let line = view.read(pos);
                let scale = if comptime!(sw > 1) {
                    line.extract_dynamic(field.fcast::<usize>())
                } else {
                    line.extract(0usize)
                };
                value * Vector::<E, V>::cast_from(scale * self.coarser.extract(0usize))
            }
            ComptimeOption::None => value,
        }
    }
}

/// Where the scale covering the value at `coords` lies: the scales' own coordinate, one entry
/// per axis of their space, its innermost a line index; and the field of that line the value's
/// coordinate falls in.
#[cube]
fn scale_coords(
    coords: &Coords<u32>,
    #[comptime] values: Space,
    #[comptime] scales: Space,
    #[comptime] sw: usize,
) -> (CoordsDyn, u32) {
    let rank = comptime!(scales.rank());
    let mut pos = CoordsDyn::new();
    let mut field = 0u32.runtime();
    #[unroll]
    for p in 0..rank {
        let axis = comptime!(scales.axis_at(p));
        let coord = coords.at(comptime!(values.position(axis)));
        if comptime!(p == rank - 1 && sw > 1) {
            field = coord.frem(comptime!(sw as u32));
            pos.push(coord.fdiv(comptime!(sw as u32)));
        } else {
            pos.push(coord);
        }
    }
    (pos, field)
}
