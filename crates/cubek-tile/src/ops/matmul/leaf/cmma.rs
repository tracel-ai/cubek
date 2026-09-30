//! The tensor-core leaf: `acc += lhs · rhs` via `cmma::execute`, with non-fragment operands
//! loaded into transient `A`/`B` fragments.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::ops::matmul::leaf::Side;
use crate::*;

#[cube]
impl<A: Numeric> CmmaData<A> {
    /// Tensor-core contraction `self += lhs · rhs`, each factor times its scales.
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(
        &self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
    ) {
        match (&lhs.kind, &rhs.kind) {
            (TileKind::PlaneTile(a), TileKind::PlaneTile(b)) => match (a, b) {
                (PlaneTile::Cmma(a), PlaneTile::Cmma(b)) => {
                    lhs.refuse_factor("PlaneTile::Mma");
                    rhs.refuse_factor("PlaneTile::Mma");
                    cmma::execute(&a.matrix, &b.matrix, &self.matrix, &self.matrix)
                }
                _ => panic!("cmma operands must be cmma fragments"),
            },
            _ => {
                let a_read = comptime!(FragmentRead::new(
                    Side::Lhs,
                    &lhs.place.space,
                    &rhs.place.space,
                    &out
                ));
                let b_read = comptime!(FragmentRead::new(
                    Side::Rhs,
                    &lhs.place.space,
                    &rhs.place.space,
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

/// How a rhs window is read: col-major when its trailing axis is `contracted`, else row-major.
pub(crate) fn rhs_layout(rhs: &Space, contracted: Axis) -> MatrixLayout {
    match rhs.axis_at(rhs.rank() - 1) == contracted {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    }
}

/// How one factor of a tensor-core contraction is read into its fragment.
#[derive(Clone)]
pub(crate) struct FragmentRead {
    /// The fragment's edges.
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// The rhs window's layout; the lhs is always row-major.
    pub layout: MatrixLayout,
    /// What this factor's scales are checked against.
    pub side: Side,
    pub out: Space,
}

impl FragmentRead {
    pub(crate) fn new(side: Side, lhs: &Space, rhs: &Space, out: &Space) -> Self {
        // A split contraction is one `k` edge.
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
impl<E: Numeric> Tile<E> {
    /// This factor read into `frag`; a scaled or landed factor goes through its landing.
    pub(crate) fn load(&self, frag: &mut Matrix<E>, #[comptime] read: FragmentRead) {
        let scaled = self.scaled();
        let landed = self.has_landing();
        let packing = self.packing();
        if comptime!(scaled || landed) {
            let landing = self.landed(comptime!(read.side), comptime!(read.out.clone()));
            landing.load_into(frag, comptime!(read.out.clone()));
            // The landing is this region's until every unit's load has read it.
            sync_plane();
        } else {
            match &self.kind {
                TileKind::Memory(m) => {
                    comptime!(assert!(
                        m.address == AddressSpace::Shared,
                        "mma: a fragment loads a window as it lies and a gmem layout is \
                         unchecked; open the operand with `with_landing()`, or stage it"
                    ));
                    comptime!(assert!(
                        packing == Packing::Plain,
                        "mma: a packed stage reaches a fragment through a landing; open the \
                         operand with `with_landing`"
                    ));
                    cmma::load(frag, m.window_slice(), m.row_stride())
                }
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("mma: an operand reaches a fragment from a memory window")
                }
            }
        }
    }

    /// This factor in its plane's landing: a dense shared-memory stage holding `values ⊗ scales`
    /// that fragments load from.
    pub(crate) fn landed(&self, #[comptime] side: Side, #[comptime] out: Space) -> Tile<E> {
        let lands = self.has_landing();
        comptime!(assert!(
            lands,
            "Tile::landed: a scaled operand reaches a tensor-core fragment through a landing in \
             shared memory; open the operand with `landed_for`"
        ));
        let space = comptime!(self.place.space.clone());
        let units = self.units();
        let planes = comptime!(plane_windows(&space, &self.place.levels));
        let (stage, mut window) = Memory::<E>::landing(comptime!(space.clone()), units, planes);
        let landing = Tile::new(stage.kind, comptime!(self.place.clone()));

        let vw = self.vector_size();
        let size!(VW) = vw;
        let view = self.nd_packed::<VW>(comptime!(Guard::Checked));
        let acc_axes = comptime!(accumulator_axes(side, &out, &space));
        let scales = self.reader(
            comptime!(MatrixAxes::trailing(&space)),
            0usize,
            side,
            comptime!(out.clone()),
            acc_axes,
        );
        let load = self.vector_tile();
        let lines = load.count(&space);
        let strides = comptime!(dense_strides(&space));
        let by_shuffle = scales.by_shuffle();
        if comptime!(by_shuffle) {
            // The whole plane must join every shuffle; units past the lines reread the last one and
            // write nothing.
            #[allow(clippy::manual_div_ceil)]
            let turns = (lines + PLANE_DIM - 1) / PLANE_DIM;
            for turn in 0..turns {
                let mine = turn * PLANE_DIM + UNIT_POS_PLANE;
                let line = min(mine, lines - 1);
                let coords = load.start(line, &space);
                let landed =
                    scales.apply_at::<E, VW>(view.read(load.index(&coords, &space)), &coords);
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
                let coords = load.start(line, &space);
                let landed =
                    scales.apply_at::<E, VW>(view.read(load.index(&coords, &space)), &coords);
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
    /// This landing loaded into `frag`, dense over the window's axes.
    pub(crate) fn load_into(&self, frag: &mut Matrix<E>, #[comptime] out: Space) {
        let cols = comptime!(landed_cols(&self.place.space, &out) as u32);
        match &self.kind {
            TileKind::Memory(m) => {
                comptime!(assert!(
                    m.address == AddressSpace::Shared,
                    "mma: a fragment loads from a shared window"
                ));
                cmma::load(frag, m.window_slice(), cols)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!("mma: a fragment loads from a shared window"),
        }
    }
}

/// The column count of a landed window's matrix: the trailing axes on the innermost axis's side
/// of the contraction.
fn landed_cols(window: &Space, out: &Space) -> usize {
    let rank = window.rank();
    let innermost = out.contains(window.axis_at(rank - 1));
    (0..rank)
        .rev()
        .take_while(|&p| out.contains(window.axis_at(p)) == innermost)
        .map(|p| window.extent_at(p))
        .product()
}

/// The accumulator's [`MatrixAxes`] read off one factor alone.
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

/// Row-major scalar strides over a dense copy of `space`.
fn dense_strides(space: &Space) -> Vec<usize> {
    let rank = space.rank();
    let mut strides = vec![1; rank];
    for p in (0..rank - 1).rev() {
        let below = strides[p + 1] * space.extent_at(p + 1);
        strides[p] = below;
    }
    strides
}

/// The scalar offset of `coords` under `strides`.
#[cube]
#[allow(clippy::needless_range_loop)]
fn offset_of(coords: &Coords<u32>, #[comptime] strides: Vec<usize>) -> usize {
    let n = coords.len();
    let mut offset = 0u32.runtime();
    #[unroll]
    for p in 0..n {
        offset = offset.plus(coords.at(p).times(comptime!(strides[p] as u32)));
    }
    offset as usize
}
