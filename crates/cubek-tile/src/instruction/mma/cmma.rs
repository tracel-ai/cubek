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
    ContractEdges, EdgeOrdinal, ScaleLevel, check_scales_ride, combined_scales,
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
    pub(crate) fn mma_scaled<L: Numeric, R: Numeric, S: Numeric>(
        &self,
        lhs: &Tile<L>,
        rhs: &Tile<R>,
        #[comptime] side: ScaleSide,
        scales: &Sequence<Tile<S>>,
        #[comptime] out: Space,
    ) {
        // The fragment's edges off the accumulator's axes and the contracted extent, as the plain
        // leaf reads them: a split contraction is one `k` edge, whatever its digits.
        let acc_axes = comptime!(MatrixAxes::accumulator(&out, &lhs.space));
        let m = comptime!(acc_axes.rows(&out));
        let n = comptime!(acc_axes.cols(&out));
        let operands = comptime!(Space::merge(&[&lhs.space, &rhs.space]));
        let k = comptime!(operands.contracted_extent(&out));
        let lrank = comptime!(lhs.space.rank());
        let contracted = comptime!(lhs.space.axis_at(lrank - 1));
        let layout = comptime!(rhs_layout(&rhs.space, contracted));
        let transposed = comptime!(layout == MatrixLayout::ColMajor);
        let lw = lhs.vector_size();
        let rw = rhs.vector_size();
        let inner = scales.index(0);
        let sw = inner.vector_size();
        comptime!(check_scales_ride(side, &inner.space, &out, acc_axes));
        // The level as the landing reads it: a line of the scaled operand takes the scale at
        // its own row and its block's column, one scale a read.
        let invariant = inner.invariant_over(comptime!(operands.clone()));
        let level = comptime!(ScaleLevel::of(
            &inner.space,
            &ContractEdges {
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
                // A col-major rhs lands lines along the contraction, one column each.
                aw: if transposed { 1 } else { rw },
                contracted_per_step: 1,
                ordinal: EdgeOrdinal::Constant,
            },
            side,
            &invariant,
            sw,
        ));
        comptime!(assert!(
            level.lanes == 1,
            "mma_scaled: the landing reads one scale a line; bind the scales one wide for the \
             fragment leaf"
        ));

        let mut a_frag =
            unsafe { Matrix::<L>::uninitialized(MatrixIdent::A, m, n, k, MatrixLayout::RowMajor) };
        let mut b_frag = unsafe { Matrix::<R>::uninitialized(MatrixIdent::B, m, n, k, layout) };
        match comptime!(side) {
            ScaleSide::Lhs => {
                let landed = land_scaled::<L, S>(lhs, scales, level, m, k, false);
                cmma::load(&mut a_frag, &landed, comptime!(k as u32));
                load_fragment(&mut b_frag, rhs);
            }
            ScaleSide::Rhs => {
                load_fragment(&mut a_frag, lhs);
                let (rows, cols) = comptime!(if transposed { (n, k) } else { (k, n) });
                let landed = land_scaled::<R, S>(rhs, scales, level, rows, cols, transposed);
                cmma::load(&mut b_frag, &landed, comptime!(cols as u32));
            }
        }
        cmma::execute(&a_frag, &b_frag, &self.matrix, &self.matrix);
        // The landing is this region's until every lane's load has read it.
        sync_plane();
    }
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
