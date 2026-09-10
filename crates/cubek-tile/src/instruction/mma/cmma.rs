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

use crate::instruction::registers::contract::{
    ContractEdges, EdgeOrdinal, Side, combined_scales, level_of,
};
use crate::instruction::registers::lines::{Lines, LinesExpand};
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
                let lw = lhs_values.vector_size();
                let rw = rhs_values.vector_size();
                let a_read = comptime!(FragmentRead::new(
                    Side::Lhs,
                    &lhs_values.space,
                    &rhs_values.space,
                    &out,
                    lw,
                    rw
                ));
                let b_read = comptime!(FragmentRead::new(
                    Side::Rhs,
                    &lhs_values.space,
                    &rhs_values.space,
                    &out,
                    lw,
                    rw
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
/// edges, the matrix this factor's window is, and what a scale level covering its lines is
/// stated against.
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
    /// This factor's own window as a matrix, and whether it is the transpose of the matrix its
    /// scales address (a rhs read col-major).
    pub rows: usize,
    pub cols: usize,
    pub transposed: bool,
    /// What a scale level over this factor is stated against.
    pub side: Side,
    pub operands: Space,
    pub out: Space,
    pub acc_axes: MatrixAxes,
    pub edges: ContractEdges,
}

impl FragmentRead {
    pub(crate) fn new(
        side: Side,
        lhs: &Space,
        rhs: &Space,
        out: &Space,
        lw: usize,
        rw: usize,
    ) -> Self {
        // The fragment's edges off the accumulator's axes and the contracted extent, as the plain
        // leaf reads them: a split contraction is one `k` edge, whatever its digits.
        let acc_axes = MatrixAxes::accumulator(out, lhs);
        let m = acc_axes.rows(out);
        let n = acc_axes.cols(out);
        let operands = Space::merge(&[lhs, rhs]);
        let k = operands.contracted_extent(out);
        let layout = rhs_layout(rhs, lhs.axis_at(lhs.rank() - 1));
        let transposed = layout == MatrixLayout::ColMajor;
        // The landing reads one scale a line, each line under its own constant ordinal.
        let edges = ContractEdges {
            mr: m,
            kc: k,
            cols: n,
            reduce: operands
                .contracting(out)
                .iter()
                .map(|&axis| (axis, operands.extent(axis)))
                .collect::<Vec<_>>(),
            columns: (acc_axes.col_split..out.rank())
                .map(|p| (out.axis_at(p), out.extent_at(p)))
                .collect::<Vec<_>>(),
            lw,
            aw: match transposed {
                true => 1,
                false => rw,
            },
            contracted_per_step: 1,
            ordinal: EdgeOrdinal::Constant,
        };
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
            transposed: transposed && side == Side::Rhs,
            side,
            operands,
            out: out.clone(),
            acc_axes,
            edges,
        }
    }
}

#[cube]
impl<E: Numeric, S: Numeric> Scaled<E, S> {
    /// This factor read into `frag`.
    ///
    /// A factor carrying no scales is read from its window as it lies, which is what a fragment
    /// load wants and why the window must be staged: a gmem layout is unchecked. One carrying
    /// scales cannot be, so it lands first ([`land`](Scaled::land)) and the load reads the
    /// landing, which is row-major by construction.
    pub(crate) fn load(&self, frag: &mut Matrix<E>, #[comptime] read: FragmentRead) {
        let levels = self.levels();
        let count = levels.len();
        if comptime!(count == 0) {
            match &self.values().tile_kind {
                TileKind::Smem(m) => cmma::load(frag, m.window_slice(), m.row_stride()),
                TileKind::Gmem(_) => panic!(
                    "mma: a fragment loads a window as it lies and a gmem layout is unchecked; \
                     stage the operand first"
                ),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_) => {
                    panic!("mma: an operand reaches a fragment from a memory window")
                }
            }
        } else {
            let landing = self.land(comptime!(read.clone()));
            cmma::load(frag, &landing, comptime!(read.cols as u32));
            // The landing is this region's until every lane's load has read it.
            sync_plane();
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

    /// This factor folded into its plane's landing as `values ⊗ scales`, row-major: each lane
    /// reads lines through the values' packed view, scales each by the block scale covering it,
    /// and writes them; the plane then syncs past the writes.
    fn land(&self, #[comptime] read: FragmentRead) -> Shared<[E]> {
        let values = self.values();
        let mut landing = match &values.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.landing(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("mma: a scaled operand lands from its memory window")
            }
        };
        let levels = self.levels();
        let level = level_of(
            &levels,
            comptime!(read.operands.clone()),
            comptime!(read.out.clone()),
            comptime!(read.acc_axes),
            comptime!(read.edges.clone()),
            comptime!(read.side),
        );
        // The landing reads one scale per line: several at a time would need each value line's
        // ordinal along the shared edge, and the landing walks its lines at runtime.
        let level = comptime!(level.expect("mma: a scaled factor carries a level"));
        comptime!(assert!(
            level.lanes == 1,
            "mma: the landing reads one scale a line; bind the scales one wide for the fragment \
             leaf"
        ));

        let vw = values.vector_size();
        let size!(VW) = vw;
        let axes = comptime!(MatrixAxes::of(&values.space, read.rows, read.cols));
        let matrix = values.matrix_packed::<VW>(axes, 0usize);
        let scale_lines = combined_scales::<S, Const<1>>(&levels, level, 0usize);

        let lps = comptime!(level.lines_per_scale as u32);
        let per_row = comptime!((read.cols / vw) as u32);
        let lines = comptime!((read.rows * read.cols / vw) as u32);
        let width = comptime!(vw as u32);
        let row_cells = comptime!(read.cols as u32);
        for i in range_stepped(UNIT_POS_PLANE, lines, PLANE_DIM) {
            let r = i / per_row;
            let c = i % per_row;
            let line = matrix.read((r, c));
            // A transposed window's line sits at the scales' column, and its row is the block.
            let scale = if comptime!(read.transposed) {
                scale_lines.line((c * width, r / lps), 0usize)
            } else {
                scale_lines.line((r, c / lps), 0usize)
            };
            let scaled = line * Vector::<E, VW>::cast_from(scale.extract(0usize));
            let base = (r * row_cells + c * width) as usize;
            #[unroll]
            for j in 0..vw {
                landing[base + j] = scaled.extract(j);
            }
        }
        sync_plane();
        landing
    }
}
