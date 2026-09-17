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
                | TileKind::Procedural(_)
                | TileKind::Chunk(_) => {
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
    /// each line's position in the values' matrix is what its scale is read at
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
            | TileKind::Procedural(_)
            | TileKind::Chunk(_) => {
                panic!("mma: an operand lands from its memory window")
            }
        };
        let vw = values.vector_size();
        let size!(VW) = vw;
        let axes = comptime!(MatrixAxes::of(&values.space, read.rows, read.cols));
        let matrix = values.matrix_packed::<VW>(axes, 0usize);
        let scales = self.lookup(
            axes,
            0usize,
            comptime!(read.side),
            comptime!(read.out.clone()),
            comptime!(read.acc_axes),
        );

        let per_row = comptime!(read.cols / vw);
        let width = comptime!(vw as u32);
        let row_cells = comptime!(read.cols as u32);
        let cells = comptime!((read.rows * per_row) as u32);
        let by_shuffle = scales.by_shuffle();
        if comptime!(by_shuffle) {
            // A scale held in the plane's lanes reaches the lane that asks by a shuffle, and a
            // shuffle is the whole plane's or nothing: every lane takes every turn, whether or
            // not a line is left for it. A lane past the lines reads the last one again and
            // writes nothing.
            #[allow(clippy::manual_div_ceil)]
            let turns = (cells + PLANE_DIM - 1) / PLANE_DIM;
            for turn in 0..turns {
                let mine = turn * PLANE_DIM + UNIT_POS_PLANE;
                let cell = min(mine, cells - 1);
                let r = cell.fdiv(comptime!(per_row as u32));
                let c = cell.frem(comptime!(per_row as u32));
                let landed = scales.apply::<E, VW>(matrix.read((r, c)), (r, c));
                if mine < cells {
                    let base = (r * row_cells + c * width) as usize;
                    #[unroll]
                    for j in 0..vw {
                        landing[base + j] = landed.extract(j);
                    }
                }
            }
        } else {
            for cell in range_stepped(UNIT_POS_PLANE, cells, PLANE_DIM) {
                let r = cell.fdiv(comptime!(per_row as u32));
                let c = cell.frem(comptime!(per_row as u32));
                let landed = scales.apply::<E, VW>(matrix.read((r, c)), (r, c));
                let base = (r * row_cells + c * width) as usize;
                #[unroll]
                for j in 0..vw {
                    landing[base + j] = landed.extract(j);
                }
            }
        }
        sync_plane();
        landing
    }
}
