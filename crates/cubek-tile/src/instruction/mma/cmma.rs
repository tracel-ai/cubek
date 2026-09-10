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
    ContractEdges, EdgeOrdinal, ScaleLevel, Side, combined_scales, level_of,
};
use crate::instruction::registers::lines::{Lines, LinesExpand};
use crate::*;

#[cube]
impl<A: Numeric> CmmaData<A> {
    /// Tensor-core contraction `self += lhs · rhs`. Fragment operands execute directly;
    /// smem operands are loaded into transient `A`/`B` fragments each call. Smem only:
    /// the engine lays stages out row-major, which `cmma::load` assumes; a gmem
    /// window's layout is unchecked here, so it must be staged first.
    pub(crate) fn mma<L: Numeric, R: Numeric>(&self, lhs: &Tile<L>, rhs: &Tile<R>) {
        match (&lhs.tile_kind, &rhs.tile_kind) {
            (TileKind::PlaneTile(a), TileKind::PlaneTile(b)) => match (a, b) {
                (PlaneTile::Cmma(a), PlaneTile::Cmma(b)) => {
                    cmma::execute(&a.matrix, &b.matrix, &self.matrix, &self.matrix)
                }
                _ => panic!("cmma operands must be cmma fragments"),
            },
            (TileKind::Smem(a), TileKind::Smem(b)) => {
                // The tile is `m × k` on lhs and `k × n` on rhs (trailing two axes; any
                // leading batch axes are extent-1 at a final tile), or `n × k` on a rhs
                // read col-major.
                let m = comptime!(lhs.space.extent_at(lhs.space.rank() - 2));
                let k = comptime!(lhs.space.extent_at(lhs.space.rank() - 1));
                let layout = comptime!(rhs_layout(
                    &rhs.space,
                    lhs.space.axis_at(lhs.space.rank() - 1)
                ));
                let n = comptime!(match layout {
                    MatrixLayout::RowMajor => rhs.space.extent_at(rhs.space.rank() - 1),
                    MatrixLayout::ColMajor => rhs.space.extent_at(rhs.space.rank() - 2),
                    MatrixLayout::Undefined => unreachable!(),
                });

                // The rendezvous with the stage fill belongs to the schedule that filled it.
                let mut a_frag = unsafe {
                    Matrix::<L>::uninitialized(MatrixIdent::A, m, n, k, MatrixLayout::RowMajor)
                };
                cmma::load(&mut a_frag, a.window_slice(), a.row_stride());
                let mut b_frag =
                    unsafe { Matrix::<R>::uninitialized(MatrixIdent::B, m, n, k, layout) };
                cmma::load(&mut b_frag, b.window_slice(), b.row_stride());

                cmma::execute(&a_frag, &b_frag, &self.matrix, &self.matrix);
            }
            _ => panic!("cmma operands must be fragments or staged smem windows"),
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

#[cube]
impl<A: Numeric> CmmaData<A> {
    /// [`mma`](CmmaData::mma) with one operand scaled: that operand's window is landed in this
    /// plane's shared memory, unpacked and scaled by the lanes, and loaded as its fragment; the
    /// other loads from its window as it stands. Marlin's shape, with the landing where Metal's
    /// fragment load wants memory.
    pub(crate) fn mma_scaled<EL: Numeric, LS: Numeric, ER: Numeric, RS: Numeric>(
        &self,
        lhs: &Scaled<EL, LS>,
        rhs: &Scaled<ER, RS>,
        #[comptime] out: Space,
    ) {
        let lhs_values = lhs.values();
        let rhs_values = rhs.values();
        let lhs_levels = lhs.levels();
        let rhs_levels = rhs.levels();
        let lhs_count = lhs_levels.len();
        let rhs_count = rhs_levels.len();
        // Which factor is landed is which one carries scales. Both at once is a quantized signal
        // against a quantized weight, and the landing would have to hold two.
        let side = comptime!(match (lhs_count > 0, rhs_count > 0) {
            (true, false) => Side::Lhs,
            (false, true) => Side::Rhs,
            _ => panic!(
                "mma: a tensor-core contraction lands one scaled factor in the plane's window; \
                 scale one side"
            ),
        });

        // The fragment's edges off the accumulator's axes and the contracted extent, as the plain
        // leaf reads them: a split contraction is one `k` edge, whatever its digits.
        let acc_axes = comptime!(MatrixAxes::accumulator(&out, &lhs_values.space));
        let m = comptime!(acc_axes.rows(&out));
        let n = comptime!(acc_axes.cols(&out));
        let operands = comptime!(Space::merge(&[&lhs_values.space, &rhs_values.space]));
        let k = comptime!(operands.contracted_extent(&out));
        let lrank = comptime!(lhs_values.space.rank());
        let contracted = comptime!(lhs_values.space.axis_at(lrank - 1));
        let layout = comptime!(rhs_layout(&rhs_values.space, contracted));
        let transposed = comptime!(layout == MatrixLayout::ColMajor);

        let lw = lhs_values.vector_size();
        let rw = rhs_values.vector_size();
        // This leaf's own edges, which is all a scale level needs of it. The landing reads one
        // scale a line, each line under its own constant ordinal.
        let edges = comptime!(ContractEdges {
            mr: m,
            kc: k,
            cols: n,
            reduce: operands
                .contracting(&out)
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
        });

        let mut a_frag =
            unsafe { Matrix::<EL>::uninitialized(MatrixIdent::A, m, n, k, MatrixLayout::RowMajor) };
        let mut b_frag = unsafe { Matrix::<ER>::uninitialized(MatrixIdent::B, m, n, k, layout) };
        match comptime!(side) {
            Side::Lhs => {
                let level = level_of(
                    &lhs_levels,
                    comptime!(operands.clone()),
                    comptime!(out.clone()),
                    comptime!(acc_axes),
                    comptime!(edges),
                    comptime!(Side::Lhs),
                );
                let landed = land_scaled::<EL, LS>(
                    &lhs_values,
                    &lhs_levels,
                    comptime!(one_scale_a_line(level)),
                    m,
                    k,
                    false,
                );
                cmma::load(&mut a_frag, &landed, comptime!(k as u32));
                load_fragment(&mut b_frag, &rhs_values);
            }
            Side::Rhs => {
                let level = level_of(
                    &rhs_levels,
                    comptime!(operands.clone()),
                    comptime!(out.clone()),
                    comptime!(acc_axes),
                    comptime!(edges),
                    comptime!(Side::Rhs),
                );
                load_fragment(&mut a_frag, &lhs_values);
                let (rows, cols) = comptime!(if transposed { (n, k) } else { (k, n) });
                let landed = land_scaled::<ER, RS>(
                    &rhs_values,
                    &rhs_levels,
                    comptime!(one_scale_a_line(level)),
                    rows,
                    cols,
                    transposed,
                );
                cmma::load(&mut b_frag, &landed, comptime!(cols as u32));
            }
        }
        cmma::execute(&a_frag, &b_frag, &self.matrix, &self.matrix);
        // The landing is this region's until every lane's load has read it.
        sync_plane();
    }
}

/// The level a landing folds, which reads one scale per line: several at a time would need each
/// value line's ordinal along the shared edge, and the landing walks its lines at runtime.
fn one_scale_a_line(level: Option<ScaleLevel>) -> ScaleLevel {
    let level = level.expect("mma_scaled: the scaled factor carries a level");
    assert!(
        level.lanes == 1,
        "mma_scaled: the landing reads one scale a line; bind the scales one wide for the \
         fragment leaf"
    );
    level
}

/// `frag` loaded from `tile`'s memory window, as the plain leaf loads it.
#[cube]
fn load_fragment<E: Numeric>(frag: &mut Matrix<E>, tile: &Tile<E>) {
    match &tile.tile_kind {
        TileKind::Gmem(m) | TileKind::Smem(m) => {
            cmma::load(frag, m.window_slice(), m.row_stride());
        }
        TileKind::PlaneTile(_)
        | TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_) => {
            panic!("mma_scaled: the unscaled operand loads its fragment from a memory window")
        }
    }
}

/// The scaled operand's window, `rows × cols` in its own order, landed row-major in its plane's
/// landing as `values ⊗ scales`: each lane reads lines through the operand's packed view, scales
/// each by the block scale covering it, and writes them; the plane then syncs past the writes.
///
/// `transposed` says the window is the transpose of the scales' matrix (a rhs read col-major),
/// so a line's scale sits at the line's column and the row's block.
#[cube]
fn land_scaled<E: Numeric, S: Numeric>(
    values: &Tile<E>,
    scales: &Sequence<Tile<S>>,
    #[comptime] level: ScaleLevel,
    #[comptime] rows: usize,
    #[comptime] cols: usize,
    #[comptime] transposed: bool,
) -> Shared<[E]> {
    let mut landing = match &values.tile_kind {
        TileKind::Gmem(g) | TileKind::Smem(g) => g.landing(),
        TileKind::PlaneTile(_)
        | TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_) => {
            panic!("mma_scaled: a scaled operand is landed from its memory window")
        }
    };
    let vw = values.vector_size();
    let size!(VW) = vw;
    let axes = comptime!(MatrixAxes::of(&values.space, rows, cols));
    let matrix = values.matrix_packed::<VW>(axes, 0usize);
    let scale_lines = combined_scales::<S, Const<1>>(scales, level, 0usize);

    let lps = comptime!(level.lines_per_scale as u32);
    let per_row = comptime!((cols / vw) as u32);
    let lines = comptime!((rows * cols / vw) as u32);
    let width = comptime!(vw as u32);
    let row_cells = comptime!(cols as u32);
    for i in range_stepped(UNIT_POS_PLANE, lines, PLANE_DIM) {
        let r = i / per_row;
        let c = i % per_row;
        let line = matrix.read((r, c));
        let scale = if comptime!(transposed) {
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
