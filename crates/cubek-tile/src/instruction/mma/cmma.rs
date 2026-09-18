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

use crate::instruction::registers::contract::Side;
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
    /// The layout the rhs window reads at: col-major where the contraction is its trailing
    /// axis and the fragment reads its window as the transpose; the lhs is always row-major.
    pub layout: MatrixLayout,
    /// What the scales riding this factor are checked against: the side they were stated on,
    /// and the accumulator.
    pub side: Side,
    pub out: Space,
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
        FragmentRead {
            m,
            n,
            k,
            layout,
            side,
            out: out.clone(),
        }
    }
}

#[cube]
impl<E: Numeric, S: Numeric> Scaled<E, S> {
    /// This factor read into `frag`.
    ///
    /// A fragment loads a window as it lies, so the window's layout has to be one the
    /// instruction can be told: a shared one is, and a global one is not. **The landing is how a
    /// factor answers that** ([`landed`](Scaled::landed)) — it holds the factor's own values in
    /// a window of shared memory the plane owns, dense over the window's axes, and the load
    /// reads them from there. A factor carrying scales has no other route, since its values do
    /// not exist anywhere until they are scaled; one carrying none takes it when it has a
    /// landing, and is read from its window as it lies when the window is already shared.
    pub(crate) fn load(&self, frag: &mut Matrix<E>, #[comptime] read: FragmentRead) {
        let values = self.values();
        let count = self.levels().len();
        let landed = values.has_landing();
        let packing = values.packing();
        if comptime!(count > 0 || landed) {
            let landing = self.landed(comptime!(read.side), comptime!(read.out.clone()));
            landing.load_into(frag, comptime!(read.out.clone()));
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
                | TileKind::Procedural(_)
                | TileKind::Lanes(_) => {
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

    /// This factor in its plane's landing: a stage in shared memory, dense over the window's
    /// own axes, holding `values ⊗ scales`, that the fragments of the window load from as they
    /// load from any shared stage. Sized here, to this window: a kernel that lands a step whole
    /// lands it once, and its partition's fragments load from it.
    ///
    /// Each lane reads lines through the values' packed view, multiplies each by the scale
    /// looked up at the line's own coordinates where the factor carries one, and writes them at
    /// those coordinates; the plane then syncs past the writes. A factor with no scales lands
    /// its values as they lie — the same walk, with nothing to multiply by — which is what
    /// makes the landing an operand's *residence* rather than a scale mechanism: it is the
    /// answer to a layout a fragment cannot be told, and a packed or scaled factor needs it for
    /// its values as well.
    ///
    /// `side` and `out` are what the scales' statement is checked against: the accumulator's
    /// space, and which factor of it this is.
    pub fn landed(&self, #[comptime] side: Side, #[comptime] out: Space) -> Tile<E> {
        let values = self.values();
        let lands = values.has_landing();
        comptime!(assert!(
            lands,
            "mma_scaled: a scaled operand reaches a tensor-core fragment through a landing in \
             shared memory; open the operand with `with_landing()`"
        ));
        let space = comptime!(values.space.clone());
        let rank = comptime!(space.rank());
        let units = values.units();
        let planes = comptime!(plane_windows(&space, &values.levels));
        let (stage, mut window) = MemData::<E>::landing(comptime!(space.clone()), units, planes);
        let landing = Tile::<E> {
            tile_kind: stage.tile_kind,
            space: comptime!(space.clone()),
            depth: comptime!(values.depth),
            levels: comptime!(values.levels.clone()),
        };

        let vw = values.vector_size();
        let size!(VW) = vw;
        let view = values.nd_packed::<VW>(comptime!(Guard::Checked));
        let acc_axes = comptime!(accumulator_axes(side, &out, &space));
        let scales = self.lookup(
            comptime!(MatrixAxes::trailing_pair(&space)),
            0usize,
            side,
            comptime!(out.clone()),
            acc_axes,
        );
        // The window's lines, the innermost axis counted in lines; and where each one lands,
        // dense over the window's axes in scalars.
        let line_extents = comptime!(line_extents(&space, vw, 0, rank));
        let lines = comptime!(line_extents.iter().product::<usize>() as u32);
        let strides = comptime!(dense_strides(&space));
        let by_shuffle = scales.by_shuffle();
        if comptime!(by_shuffle) {
            // A scale held in the plane's lanes reaches the lane that asks by a shuffle, and a
            // shuffle is the whole plane's or nothing: every lane takes every turn, whether or
            // not a line is left for it. A lane past the lines reads the last one again and
            // writes nothing.
            #[allow(clippy::manual_div_ceil)]
            let turns = (lines + PLANE_DIM - 1) / PLANE_DIM;
            for turn in 0..turns {
                let mine = turn * PLANE_DIM + UNIT_POS_PLANE;
                let line = min(mine, lines - 1);
                let coords = coords_of_line(line, comptime!(line_extents.clone()), vw);
                let landed = scales.apply_at::<E, VW>(view.read(as_dyn(&coords, vw)), &coords);
                if mine < lines {
                    let base = offset_of(&coords, comptime!(strides.clone()));
                    #[unroll]
                    for j in 0..vw {
                        window[base + j] = landed.extract(j);
                    }
                }
            }
        } else {
            for line in range_stepped(UNIT_POS_PLANE, lines, PLANE_DIM) {
                let coords = coords_of_line(line, comptime!(line_extents.clone()), vw);
                let landed = scales.apply_at::<E, VW>(view.read(as_dyn(&coords, vw)), &coords);
                let base = offset_of(&coords, comptime!(strides.clone()));
                #[unroll]
                for j in 0..vw {
                    window[base + j] = landed.extract(j);
                }
            }
        }
        sync_plane();
        landing
    }
}

#[cube]
impl<E: Numeric> Tile<E> {
    /// This landing loaded into `frag` as it lies: dense over the window's axes, so a row of
    /// the fragment's matrix is as long as the columns' axes multiply to — the trailing run of
    /// axes on the innermost one's side of the contraction against `out`.
    pub(crate) fn load_into(&self, frag: &mut Matrix<E>, #[comptime] out: Space) {
        let cols = comptime!(landed_cols(&self.space, &out) as u32);
        match &self.tile_kind {
            TileKind::Smem(m) => cmma::load(frag, m.window_slice(), cols),
            TileKind::Gmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => panic!("mma: a fragment loads from a shared window"),
        }
    }
}

/// The columns of a landed window's matrix: the product of its trailing axes on the innermost
/// axis's side of the contraction — contracted with it, or the accumulator's with it — which is
/// where a row-major copy of the window turns to its next row.
fn landed_cols(window: &Space, out: &Space) -> usize {
    let rank = window.rank();
    let innermost = out.contains(window.axis_at(rank - 1));
    (0..rank)
        .rev()
        .take_while(|&p| out.contains(window.axis_at(p)) == innermost)
        .map(|p| window.extent_at(p))
        .product()
}

/// The accumulator's matrix as [`MatrixAxes::accumulator`] reads it off the lhs, read off one
/// factor alone: the column group is the innermost run of `out`'s axes the lhs does not span,
/// which is the run the rhs does.
fn accumulator_axes(side: Side, out: &Space, own: &Space) -> MatrixAxes {
    let mut col_split = out.rank() - 1;
    let in_columns = |axis: Axis| match side {
        Side::Lhs => !own.contains(axis),
        Side::Rhs => own.contains(axis),
    };
    while col_split > 1 && in_columns(out.axis_at(col_split - 1)) {
        col_split -= 1;
    }
    MatrixAxes {
        row_split: col_split - 1,
        col_split,
    }
}

/// Row-major scalar strides over `space`: where a coordinate lands in a dense copy of it.
fn dense_strides(space: &Space) -> Vec<usize> {
    let rank = space.rank();
    let mut strides = vec![1; rank];
    for p in (0..rank - 1).rev() {
        let below = strides[p + 1] * space.extent_at(p + 1);
        strides[p] = below;
    }
    strides
}

/// The scalar coordinate of the `line`-th line of a window whose innermost axis counts in
/// `vw`-wide lines: one entry per axis, the line's first value.
#[cube]
fn coords_of_line(
    line: u32,
    #[comptime] line_extents: Vec<usize>,
    #[comptime] vw: usize,
) -> Coords<u32> {
    let n = comptime!(line_extents.len());
    let digits = unravel_const(line_extents, line);
    let mut coords = Coords::<u32>::new();
    #[unroll]
    for p in 0..n {
        if comptime!(p == n - 1) {
            coords.push(digits.at(p).fmul(comptime!(vw as u32)));
        } else {
            coords.push(digits.at(p));
        }
    }
    coords
}

/// `coords` as an N-D view addresses them: the innermost a line index.
#[cube]
fn as_dyn(coords: &Coords<u32>, #[comptime] vw: usize) -> CoordsDyn {
    let n = coords.len();
    let mut at = CoordsDyn::new();
    #[unroll]
    for p in 0..n {
        if comptime!(p == n - 1) {
            at.push(coords.at(p).fdiv(comptime!(vw as u32)));
        } else {
            at.push(coords.at(p));
        }
    }
    at
}

/// Where `coords` lands in a dense copy: the scalar offset under `strides`.
#[cube]
#[allow(clippy::needless_range_loop)]
fn offset_of(coords: &Coords<u32>, #[comptime] strides: Vec<usize>) -> usize {
    let n = coords.len();
    let mut offset = 0u32.runtime();
    #[unroll]
    for p in 0..n {
        offset = offset.fadd(coords.at(p).fmul(comptime!(strides[p] as u32)));
    }
    offset as usize
}
