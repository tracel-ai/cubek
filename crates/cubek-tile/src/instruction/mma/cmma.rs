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

use crate::instruction::registers::contract::{ContractEdges, Side, combined_scales, level_of};
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
        if comptime!(count > 0 || landed) {
            let landing = match comptime!(count > 0) {
                true => self.land(comptime!(read.clone())),
                false => self.land_plain(comptime!(read.clone())),
            };
            cmma::load(frag, &landing, comptime!(read.cols as u32));
            // The landing is this region's until every lane's load has read it.
            sync_plane();
        } else {
            match &values.tile_kind {
                TileKind::Smem(m) => cmma::load(frag, m.window_slice(), m.row_stride()),
                TileKind::Gmem(_) => panic!(
                    "mma: a fragment loads a window as it lies and a gmem layout is unchecked; \
                     open the operand with `with_landing(planes, lanes)`, or stage it"
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
    /// packed view, multiplies each by the block scale covering it where the factor carries
    /// one, and writes them; the plane then syncs past the writes.
    ///
    /// A factor with no scales lands its values as they lie, which is what makes the landing an
    /// operand's *residence* rather than a scale mechanism: it is the answer to a layout a
    /// fragment cannot be told, and a packed or scaled factor needs it for its values as well.
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
        let level = comptime!(level.expect("mma: a scaled factor carries a level"));

        let vw = values.vector_size();
        let size!(VW) = vw;
        let size!(SW) = comptime!(level.lanes);
        let axes = comptime!(MatrixAxes::of(&values.space, read.rows, read.cols));
        let matrix = values.matrix_packed::<VW>(axes, 0usize);
        let scale_lines = combined_scales::<S, SW>(&levels, level, 0usize);

        // The plane deals the coordinate a scale is constant along — the row, or the line column
        // of a transposed window — and a lane steps the scale lines of its share, building under
        // each the `fields · lines` value lines it covers. One read of the scales serves all of
        // them, every field of the word used.
        let fields = comptime!(level.lanes);
        let lps = comptime!(level.lines_per_scale);
        let per_scale_line = comptime!(fields * lps);
        let per_row = comptime!(read.cols / vw);
        let width = comptime!(vw as u32);
        let row_cells = comptime!(read.cols as u32);
        if comptime!(read.transposed) {
            // A transposed window's line sits at the scales' column, and its row is the block.
            comptime!(assert!(
                read.rows.is_multiple_of(per_scale_line),
                "mma: {} rows of a transposed landing are not whole scale lines of {per_scale_line}",
                read.rows
            ));
            for c in range_stepped(UNIT_POS_PLANE, comptime!(per_row as u32), PLANE_DIM) {
                for s in 0..comptime!(read.rows / per_scale_line) {
                    let scale = scale_lines.line((c * width, s as u32));
                    #[unroll]
                    for field in 0..fields {
                        for l in 0..lps {
                            let r =
                                (s * comptime!(per_scale_line) + comptime!(field * lps) + l) as u32;
                            let scaled = matrix.read((r, c))
                                * Vector::<E, VW>::cast_from(scale.extract(field));
                            let base = (r * row_cells + c * width) as usize;
                            #[unroll]
                            for j in 0..vw {
                                landing[base + j] = scaled.extract(j);
                            }
                        }
                    }
                }
            }
        } else {
            comptime!(assert!(
                per_row.is_multiple_of(per_scale_line),
                "mma: {per_row} lines of a landing's row are not whole scale lines of \
                 {per_scale_line}"
            ));
            for r in range_stepped(UNIT_POS_PLANE, comptime!(read.rows as u32), PLANE_DIM) {
                for s in 0..comptime!(per_row / per_scale_line) {
                    let scale = scale_lines.line((r, s as u32));
                    #[unroll]
                    for field in 0..fields {
                        for l in 0..lps {
                            let c =
                                (s * comptime!(per_scale_line) + comptime!(field * lps) + l) as u32;
                            let scaled = matrix.read((r, c))
                                * Vector::<E, VW>::cast_from(scale.extract(field));
                            let base = (r * row_cells + c * width) as usize;
                            #[unroll]
                            for j in 0..vw {
                                landing[base + j] = scaled.extract(j);
                            }
                        }
                    }
                }
            }
        }
        sync_plane();
        landing
    }

    /// [`land`](Scaled::land) for a factor that carries no scales: the same walk, writing each
    /// line as it lies. What the landing buys here is the layout alone — a window the fragment
    /// load can be told the stride of.
    fn land_plain(&self, #[comptime] read: FragmentRead) -> Shared<[E]> {
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
        let vw = values.vector_size();
        let size!(VW) = vw;
        let axes = comptime!(MatrixAxes::of(&values.space, read.rows, read.cols));
        let matrix = values.matrix_packed::<VW>(axes, 0usize);

        let per_row = comptime!((read.cols / vw) as u32);
        let lines = comptime!((read.rows * read.cols / vw) as u32);
        let width = comptime!(vw as u32);
        let row_cells = comptime!(read.cols as u32);
        for i in range_stepped(UNIT_POS_PLANE, lines, PLANE_DIM) {
            let r = i / per_row;
            let c = i % per_row;
            let line = matrix.read((r, c));
            let base = (r * row_cells + c * width) as usize;
            #[unroll]
            for j in 0..vw {
                landing[base + j] = line.extract(j);
            }
        }
        sync_plane();
        landing
    }
}
