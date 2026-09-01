//! Attention's matmul leaves at fragment ownership: the plane is the worker, and each plane
//! contracts every `planes`-th fragment of the output through the hardware instruction the space
//! states. The [`columns`](super::columns) twin runs the same two matmuls at unit ownership under
//! the software instruction; which one a call reaches is the space's
//! [`instruction`](crate::Space::instruction) and nothing else.
//!
//! Nothing lives in a fragment across a step. Each visit loads its operands, contracts, and stores
//! the result back to shared memory, so the scalar work between two matmuls (the mask probe, the
//! rescale) meets no fragment in flight: this needs no fragment arithmetic and no accumulator
//! surviving a barrier.
//!
//! The visit box is the space's own statement. The accumulator's instruction level gives the
//! fragment's `m × n` and the lhs's level its contraction depth `k`; the rhs is cut by the grid
//! those two imply, since its own space describes the block the caller's walk handed over, not the
//! fragments inside it.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
};

use super::visit_box;
use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// [`score`](Tile::score) under a hardware instruction: `self[r, c] = dot(q[r, :], k[c, :])`,
    /// one fragment per visit.
    ///
    /// `k` is read through a col-major operand fragment, so the score contracts `k`'s trailing
    /// head dim against `q`'s with no transposed copy: the fragment *is* the `k × n` matrix its
    /// space says it is, read out of the `{cols, k}` window it was stored as.
    ///
    /// Fragments whose columns all lie at or past `cols_bound` are skipped, their cells left to
    /// the softmax's mask probe. A fragment straddling the bound is computed whole, so the block
    /// the caller hands over must be readable to its edge.
    pub(crate) fn score_fragments<EI: Numeric>(
        &mut self,
        q: &Tile<EI>,
        k: &Tile<EI>,
        cols_bound: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let cols = comptime!(self.space.extent_at(1));
        let d = comptime!(q.space.extent_at(1));
        let (m, n) = comptime!(visit_box(&self.space));
        let (qm, kc) = comptime!(visit_box(&q.space));
        comptime!(assert_fragment_shapes(
            "score",
            (&self.space, &q.space, &k.space),
            (rows, cols, d),
            (m, n, kc),
            qm,
            (cols, d),
        ));

        let row = comptime!(self.space.axis_at(0));
        let col = comptime!(self.space.axis_at(1));
        let con = comptime!(q.space.axis_at(1));
        let shape = comptime!((m, n, kc));
        let acc_form = comptime!(Space::new(&[(row, m), (col, n)]));
        let lhs_form = comptime!(Space::new(&[(row, m), (con, kc)]));
        // The col-major operand read of a `{cols, k}` window: `k × n`, which is what the
        // contraction's edges have to line up with.
        let rhs_form = comptime!(Space::new(&[(con, kc), (col, n)]));

        let out = self.retiled(comptime!(tiled(&self.space, m, n)));
        let q = q.retiled(comptime!(tiled(&q.space, m, kc)));
        let k = k.retiled(comptime!(tiled(&k.space, n, kc)));

        let grid_c = comptime!(cols / n);
        let steps = comptime!(d / kc);
        let visits = comptime!((rows / m) * grid_c);
        // Never zero: a cube narrower than a plane cannot run a plane instruction at all, and a
        // step of zero would be a hang rather than a refusal.
        let planes = max(CUBE_DIM as usize / PLANE_DIM as usize, 1usize);
        let mut visit = UNIT_POS as usize / PLANE_DIM as usize;
        while visit < visits {
            let fr = visit / grid_c;
            let fc = visit % grid_c;
            if fc * n < cols_bound {
                let mut acc = fragment::<EA>(
                    MatrixIdent::Accumulator,
                    MatrixLayout::RowMajor,
                    shape,
                    comptime!(acc_form.clone()),
                );
                acc.zero();
                for step in 0..steps {
                    let mut lhs = fragment::<EI>(
                        MatrixIdent::A,
                        MatrixLayout::RowMajor,
                        shape,
                        comptime!(lhs_form.clone()),
                    );
                    lhs.copy_from(&q.at(&window(comptime!(q.space.clone()), fr, step)));
                    let mut rhs = fragment::<EI>(
                        MatrixIdent::B,
                        MatrixLayout::ColMajor,
                        shape,
                        comptime!(rhs_form.clone()),
                    );
                    rhs.copy_from(&k.at(&window(comptime!(k.space.clone()), fc, step)));
                    acc.mma(&lhs, &rhs, comptime!(Semiring::SUM_PROD));
                }
                let mut cell = out.at(&window(comptime!(out.space.clone()), fr, fc));
                cell.copy_from(&acc);
            }
            visit += planes;
        }
    }

    /// [`mix`](Tile::mix) under a hardware instruction:
    /// `self[r, :] = self[r, :] · factors[r] + Σ_{c < cols_bound} p[r, c] · val[c, :]`.
    ///
    /// The rescale runs where the running total lies, cube-wide, before any fragment is loaded:
    /// no elementwise math touches a fragment, so the accumulator is scaled in shared memory and
    /// read back as an accumulator fragment, which is the one sync this leaf owns. The caller
    /// syncs on both sides as before.
    ///
    /// Contraction steps at or past `cols_bound` are skipped: stale cache beyond the attended
    /// prefix (possibly NaN) must not ride a zero probability. A step straddling the bound is
    /// contracted whole, so the block must be readable to its edge.
    pub(crate) fn mix_fragments<EP: Numeric, EI: Numeric>(
        &mut self,
        p: &Tile<EP>,
        val: &Tile<EI>,
        factors: &Tile<EA>,
        cols_bound: usize,
    ) {
        let rows = comptime!(self.space.extent_at(0));
        let val_dim = comptime!(self.space.extent_at(1));
        let cols = comptime!(p.space.extent_at(1));
        let (m, n) = comptime!(visit_box(&self.space));
        let (pm, kc) = comptime!(visit_box(&p.space));
        comptime!(assert_fragment_shapes(
            "mix",
            (&self.space, &p.space, &val.space),
            (rows, val_dim, cols),
            (m, n, kc),
            pm,
            (cols, val_dim),
        ));

        let row = comptime!(self.space.axis_at(0));
        let col = comptime!(self.space.axis_at(1));
        let con = comptime!(p.space.axis_at(1));
        let shape = comptime!((m, n, kc));
        let acc_form = comptime!(Space::new(&[(row, m), (col, n)]));
        let lhs_form = comptime!(Space::new(&[(row, m), (con, kc)]));
        let rhs_form = comptime!(Space::new(&[(con, kc), (col, n)]));

        self.scale_rows(factors);
        sync_cube();

        let out = self.retiled(comptime!(tiled(&self.space, m, n)));
        let p = p.retiled(comptime!(tiled(&p.space, m, kc)));
        let val = val.retiled(comptime!(tiled(&val.space, kc, n)));

        let grid_c = comptime!(val_dim / n);
        let steps = comptime!(cols / kc);
        let visits = comptime!((rows / m) * grid_c);
        // Never zero: a cube narrower than a plane cannot run a plane instruction at all, and a
        // step of zero would be a hang rather than a refusal.
        let planes = max(CUBE_DIM as usize / PLANE_DIM as usize, 1usize);
        let mut visit = UNIT_POS as usize / PLANE_DIM as usize;
        while visit < visits {
            let fr = visit / grid_c;
            let fc = visit % grid_c;
            let mut acc = fragment::<EA>(
                MatrixIdent::Accumulator,
                MatrixLayout::RowMajor,
                shape,
                comptime!(acc_form.clone()),
            );
            let mut cell = out.at(&window(comptime!(out.space.clone()), fr, fc));
            acc.copy_from(&cell);
            for step in 0..steps {
                if step * kc < cols_bound {
                    let mut lhs = fragment::<EP>(
                        MatrixIdent::A,
                        MatrixLayout::RowMajor,
                        shape,
                        comptime!(lhs_form.clone()),
                    );
                    lhs.copy_from(&p.at(&window(comptime!(p.space.clone()), fr, step)));
                    let mut rhs = fragment::<EI>(
                        MatrixIdent::B,
                        MatrixLayout::RowMajor,
                        shape,
                        comptime!(rhs_form.clone()),
                    );
                    rhs.copy_from(&val.at(&window(comptime!(val.space.clone()), step, fc)));
                    acc.mma(&lhs, &rhs, comptime!(Semiring::SUM_PROD));
                }
            }
            cell.copy_from(&acc);
            visit += planes;
        }
    }
}

/// One uninitialized fragment of the instruction's `(m, n, k)` shape, in the role and layout its
/// operand is read at. `space` is the matrix it *is*, which the contraction's edges line up with:
/// for a col-major operand that is the transpose of the window it was stored as.
#[cube]
fn fragment<E: Numeric>(
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] shape: (usize, usize, usize),
    #[comptime] space: Space,
) -> Tile<E> {
    let (m, n, k) = comptime!(shape);
    CmmaData::<E>::fragment(ident, m, n, k, layout, space)
}

/// The region at `(r, c)` of a rank-2 space, coordinates runtime: the visit a plane picked out
/// of the grid, which no [`Walk`](crate::Walk) enumerates because the workers here are planes.
#[cube]
fn window(#[comptime] space: Space, r: usize, c: usize) -> Region {
    let mut coords = Coords::<u32>::new();
    coords.push(r as u32);
    coords.push(c as u32);
    Region::new(coords, space)
}

/// `space` cut into `e0 × e1` windows, whatever it stated: the fragment grid is one fact, and a
/// leaf that reads it off the accumulator applies it to every operand.
fn tiled(space: &Space, e0: usize, e1: usize) -> Space {
    let (a0, a1) = (space.axis_at(0), space.axis_at(1));
    Tiling::new()
        .extents(&[(a0, space.extent_at(0)), (a1, space.extent_at(1))])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(a0, Cut::sequential(e0))
                .axis(a1, Cut::sequential(e1))
        })
        .build()
}

/// What a fragment leaf needs of its three operands: rank-2 tiles whose grids divide, the lhs
/// meeting the accumulator's rows, and the rhs holding the `(cols, k)` the other two imply.
fn assert_fragment_shapes(
    verb: &str,
    spaces: (&Space, &Space, &Space),
    extents: (usize, usize, usize),
    fragment: (usize, usize, usize),
    lhs_rows: usize,
    rhs_extents: (usize, usize),
) {
    let (out, lhs, rhs) = spaces;
    let (rows, cols, contracted) = extents;
    let (m, n, kc) = fragment;
    assert!(
        out.rank() == 2 && lhs.rank() == 2 && rhs.rank() == 2,
        "{verb}: a fragment leaf reads rank-2 tiles ({out:?}, {lhs:?}, {rhs:?}); stage the rows \
         as one axis"
    );
    assert!(
        lhs.extent_at(0) == rows && lhs.extent_at(1) == contracted,
        "{verb}: the lhs is {lhs:?}, not the {rows}x{contracted} this accumulator contracts"
    );
    assert!(
        rhs.extent_at(0) == rhs_extents.0 && rhs.extent_at(1) == rhs_extents.1,
        "{verb}: the rhs is {rhs:?}, not the {}x{} this accumulator contracts",
        rhs_extents.0,
        rhs_extents.1
    );
    assert!(
        lhs_rows == m,
        "{verb}: the lhs visits {lhs_rows} rows and the accumulator {m}; both levels cut the \
         same rows"
    );
    assert!(
        rows.is_multiple_of(m) && cols.is_multiple_of(n) && contracted.is_multiple_of(kc),
        "{verb}: the {m}x{n}x{kc} fragment does not tile a {rows}x{cols} accumulator contracting \
         {contracted}"
    );
}
