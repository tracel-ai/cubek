//! Attention's matmul leaves at fragment ownership: the plane is the worker, and each plane
//! contracts every `planes`-th fragment of the output through the hardware instruction the space
//! states. The [`columns`](super::columns) twin runs the same two matmuls at unit ownership under
//! the software instruction; which one a call reaches is the space's
//! [`instruction`](crate::Space::instruction) and nothing else.
//!
//! Nothing lives in a fragment across a step. Each visit loads its operands, contracts, and stores
//! the result back to shared memory, so the scalar work between two matmuls (the mask probe, the
//! rescale) meets no fragment in flight: this leaf needs no fragment arithmetic and no accumulator
//! surviving a barrier.

use core::fmt::{self, Display, Formatter};

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

#[cube]
impl<EA: Float> Tile<EA> {
    /// [`score`](Tile::score) under a hardware instruction: `self[r, c] = dot(q[r, :], k[c, :])`,
    /// one fragment per visit.
    ///
    /// `k` is read through a col-major operand fragment, so the score contracts `k`'s trailing
    /// head dim against `q`'s with no transposed copy.
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
        let mma = comptime!(Contraction::of(
            Matmul::Score,
            &self.space,
            &q.space,
            &k.space
        ));
        let (m, n, kc) = comptime!((mma.m, mma.n, mma.k));
        let cols = comptime!(self.space.extent_at(1));
        let head_dim = comptime!(q.space.extent_at(1));
        let out = self.retiled(comptime!(tiled(&self.space, m, n)));
        let q = q.retiled(comptime!(tiled(&q.space, m, kc)));
        let k = k.retiled(comptime!(tiled(&k.space, n, kc)));

        let grid = comptime!(cols / n);
        let steps = comptime!(head_dim / kc);
        let visits = comptime!((self.space.extent_at(0) / m) * grid);
        // Each plane takes every `planes`-th fragment, from wherever the launch put it in the
        // cube. The stride is never zero: a cube narrower than a plane cannot run a plane
        // instruction at all, and a zero step would hang rather than refuse.
        let planes = max(CUBE_DIM as usize / PLANE_DIM as usize, 1usize);
        let mut visit = UNIT_POS as usize / PLANE_DIM as usize;
        while visit < visits {
            let row = visit / grid;
            let col = visit % grid;
            if col * n < cols_bound {
                let mut acc = fragment::<EA>(MatrixIdent::Accumulator, comptime!(mma.clone()));
                acc.zero();
                for step in 0..steps {
                    let mut lhs = fragment::<EI>(MatrixIdent::A, comptime!(mma.clone()));
                    let window = Region::trailing(comptime!(q.space.clone()), row, step);
                    lhs.copy_from(&q.at(&window));
                    let mut rhs = fragment::<EI>(MatrixIdent::B, comptime!(mma.clone()));
                    let window = Region::trailing(comptime!(k.space.clone()), col, step);
                    rhs.copy_from(&k.at(&window));
                    acc.mma(&lhs, &rhs, comptime!(Semiring::SUM_PROD));
                }
                let window = Region::trailing(comptime!(out.space.clone()), row, col);
                let mut cell = out.at(&window);
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
    /// syncs on both sides.
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
        let mma = comptime!(Contraction::of(
            Matmul::Mix,
            &self.space,
            &p.space,
            &val.space
        ));
        let (m, n, kc) = comptime!((mma.m, mma.n, mma.k));
        let val_dim = comptime!(self.space.extent_at(1));
        let cols = comptime!(p.space.extent_at(1));

        self.scale_rows(factors);
        sync_cube();

        let out = self.retiled(comptime!(tiled(&self.space, m, n)));
        let p = p.retiled(comptime!(tiled(&p.space, m, kc)));
        let val = val.retiled(comptime!(tiled(&val.space, kc, n)));

        let grid = comptime!(val_dim / n);
        let steps = comptime!(cols / kc);
        let visits = comptime!((self.space.extent_at(0) / m) * grid);
        // Every `planes`-th fragment, as in `score_fragments`.
        let planes = max(CUBE_DIM as usize / PLANE_DIM as usize, 1usize);
        let mut visit = UNIT_POS as usize / PLANE_DIM as usize;
        while visit < visits {
            let row = visit / grid;
            let col = visit % grid;
            let mut acc = fragment::<EA>(MatrixIdent::Accumulator, comptime!(mma.clone()));
            let window = Region::trailing(comptime!(out.space.clone()), row, col);
            let mut cell = out.at(&window);
            acc.copy_from(&cell);
            for step in 0..steps {
                if step * kc < cols_bound {
                    let mut lhs = fragment::<EP>(MatrixIdent::A, comptime!(mma.clone()));
                    let window = Region::trailing(comptime!(p.space.clone()), row, step);
                    lhs.copy_from(&p.at(&window));
                    let mut rhs = fragment::<EI>(MatrixIdent::B, comptime!(mma.clone()));
                    let window = Region::trailing(comptime!(val.space.clone()), step, col);
                    rhs.copy_from(&val.at(&window));
                    acc.mma(&lhs, &rhs, comptime!(Semiring::SUM_PROD));
                }
            }
            cell.copy_from(&acc);
            visit += planes;
        }
    }
}

/// Which of the fold's two matmuls a contraction is. It names the leaf in the refusals, and it
/// decides how the rhs operand is read: the score contracts the keys transposed, the mix reads the
/// values as they lie.
#[derive(Clone, Copy, Debug)]
enum Matmul {
    Score,
    Mix,
}

impl Matmul {
    /// The matrix the rhs operand fragment is read as.
    fn rhs_layout(self) -> MatrixLayout {
        match self {
            Matmul::Score => MatrixLayout::ColMajor,
            Matmul::Mix => MatrixLayout::RowMajor,
        }
    }
}

impl Display for Matmul {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Matmul::Score => write!(f, "score"),
            Matmul::Mix => write!(f, "mix"),
        }
    }
}

/// The contraction one visit runs: the fragment's `m × n × k`, and the matrix each role is read
/// as at it.
///
/// The accumulator's instruction level states `m × n` and the lhs's level the depth `k`; the rhs
/// states nothing, being the block a caller's walk handed over rather than the fragments inside
/// it. Which matrix a fragment *is* can differ from the window it is read from — a col-major
/// operand is the transpose of its window — and it is the matrix that has to line up with the
/// contraction's edges.
#[derive(Clone, Debug)]
struct Contraction {
    matmul: Matmul,
    m: usize,
    n: usize,
    k: usize,
    acc: Space,
    lhs: Space,
    rhs: Space,
}

impl Contraction {
    /// The contraction `out += lhs · rhs` the three spaces state, refusing any extent they do not
    /// agree on. `matmul` names the leaf in the refusals and says how the rhs's window is read:
    /// col-major from a `{cols, k}` window, row-major from a `{k, cols}` one.
    fn of(matmul: Matmul, out: &Space, lhs: &Space, rhs: &Space) -> Self {
        let layout = matmul.rhs_layout();
        assert!(
            out.rank() == 2 && lhs.rank() == 2 && rhs.rank() == 2,
            "{matmul}: a fragment leaf reads rank-2 tiles ({out:?}, {lhs:?}, {rhs:?}); stage the \
             rows as one axis"
        );
        let (rows, cols) = (out.extent_at(0), out.extent_at(1));
        let contracted = lhs.extent_at(1);
        // A visit covers the box the level the instruction sits on cuts, or the whole tile where
        // that level cut nothing.
        let (acc_box, lhs_box) = (out.sub_tile_space(), lhs.sub_tile_space());
        let (m, n) = (acc_box.extent_at(0), acc_box.extent_at(1));
        let (lhs_rows, k) = (lhs_box.extent_at(0), lhs_box.extent_at(1));
        let stored = match layout {
            MatrixLayout::ColMajor => (cols, contracted),
            MatrixLayout::RowMajor => (contracted, cols),
            MatrixLayout::Undefined => {
                panic!("{matmul}: an operand fragment is read row- or col-major")
            }
        };
        assert!(
            lhs.extent_at(0) == rows,
            "{matmul}: the lhs is {lhs:?}, not the {rows}x{contracted} this accumulator contracts"
        );
        assert!(
            (rhs.extent_at(0), rhs.extent_at(1)) == stored,
            "{matmul}: the rhs is {rhs:?}, not the {}x{} window a {layout:?} operand is read \
             from here",
            stored.0,
            stored.1
        );
        assert!(
            lhs_rows == m,
            "{matmul}: the lhs visits {lhs_rows} rows and the accumulator {m}; both levels cut \
             the same rows"
        );
        assert!(
            rows.is_multiple_of(m) && cols.is_multiple_of(n) && contracted.is_multiple_of(k),
            "{matmul}: the {m}x{n}x{k} fragment does not tile a {rows}x{cols} accumulator \
             contracting {contracted}"
        );
        // The contracted axis is the lhs's own label: the rhs may carry another for the same axis
        // (a value block's positions are the score's columns), and the edges have to be one thing.
        let (row, col, con) = (out.axis_at(0), out.axis_at(1), lhs.axis_at(1));
        Contraction {
            matmul,
            m,
            n,
            k,
            acc: Space::new(&[(row, m), (col, n)]),
            lhs: Space::new(&[(row, m), (con, k)]),
            rhs: Space::new(&[(con, k), (col, n)]),
        }
    }

    /// The matrix `role`'s fragment is.
    fn form(&self, role: MatrixIdent) -> Space {
        match role {
            MatrixIdent::Accumulator => self.acc.clone(),
            MatrixIdent::A => self.lhs.clone(),
            MatrixIdent::B => self.rhs.clone(),
        }
    }

    /// How `role`'s fragment is read. Only the rhs is ever transposed, and which way is the
    /// matmul's own statement, so no call site repeats it.
    fn layout(&self, role: MatrixIdent) -> MatrixLayout {
        match role {
            MatrixIdent::Accumulator | MatrixIdent::A => MatrixLayout::RowMajor,
            MatrixIdent::B => self.matmul.rhs_layout(),
        }
    }
}

/// One uninitialized fragment of `mma`'s shape in `role`, read as `mma` states.
#[cube]
fn fragment<E: Numeric>(#[comptime] role: MatrixIdent, #[comptime] mma: Contraction) -> Tile<E> {
    let (m, n, k) = comptime!((mma.m, mma.n, mma.k));
    CmmaData::<E>::fragment(
        role,
        m,
        n,
        k,
        comptime!(mma.layout(role)),
        comptime!(mma.form(role)),
    )
}

/// `space` cut into `e0 × e1` windows, whatever it stated: the fragment grid is one fact, and a
/// leaf that reads it off the accumulator applies it to every operand.
fn tiled(space: &Space, e0: usize, e1: usize) -> Space {
    let (a0, a1) = (space.axis_at(0), space.axis_at(1));
    Tiling::over(
        &mut (),
        &[(a0, space.extent_at(0)), (a1, space.extent_at(1))],
    )
    .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
        l.walk(&[(a0, e0), (a1, e1)]);
    })
    .build()
}
