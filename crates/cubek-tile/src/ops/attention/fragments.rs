//! Attention's matmul leaves at fragment ownership: the plane is the worker, and each plane
//! contracts every `planes`-th fragment of the output through the hardware instruction the space
//! states. The [`columns`](super::columns) twin runs the same two matmuls at unit ownership under
//! the software instruction; which one a call reaches is the space's
//! [`instruction`](crate::Space::instruction) and nothing else.
//!
//! Trailing-two-axes convention (the matmul's, and [`columns`](super::columns)'s): the two axes a
//! matmul contracts are each tile's *last* two, and an axis above them is one the caller's walk has
//! already fixed — a batch, a KV head. Those must span a single position here, because a fragment
//! steps its rows by one physical stride and a [`Region`](crate::Region) puts a zero on them.
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
        let cols = comptime!(trailing(&self.space).1);
        let head_dim = comptime!(trailing(&q.space).1);
        let steps = comptime!(head_dim / kc);

        let q = q.retiled(comptime!(tiled(&q.space, m, kc)));

        match comptime!(plane_slice(&self.space)) {
            // The space states each plane's slice of the score: `{rows, cols / planes}`, a run
            // of key positions. The plane holds its slice's fragments across the head dim, so
            // each Q fragment is loaded once per step and reused across the slice's columns.
            Some(slice) => {
                let planes = comptime!(cols / trailing(&slice).1);
                let plane = UNIT_POS as usize / PLANE_DIM as usize;
                let mine = self.at(&Region::trailing(
                    comptime!(self.space.clone()),
                    0usize,
                    plane,
                ));
                let out = mine.retiled(comptime!(tiled(&mine.space, m, n)));
                // The keys this slice scores: its own positions, every head-dim step. The
                // positions are the window's *leading* axis, unlike the values' trailing one.
                let k = k.retiled(comptime!(tiled(&k.space, cols / planes, head_dim)));
                let k = k.at(&Region::trailing(comptime!(k.space.clone()), plane, 0usize));
                let k = k.retiled(comptime!(tiled(&k.space, n, kc)));

                let rm = comptime!(trailing(&slice).0 / m);
                let cn = comptime!(trailing(&slice).1 / n);
                let mut accs = Sequence::<Tile<EA>>::new();
                #[unroll]
                for _ in 0..comptime!(rm * cn) {
                    let mut acc = fragment::<EA>(MatrixIdent::Accumulator, comptime!(mma.clone()));
                    acc.zero();
                    accs.push(acc);
                }
                #[unroll]
                for step in 0..steps {
                    #[unroll]
                    for r in 0..rm {
                        let mut lhs = fragment::<EI>(MatrixIdent::A, comptime!(mma.clone()));
                        lhs.copy_from(&q.at(&Region::trailing(
                            comptime!(q.space.clone()),
                            r,
                            step,
                        )));
                        #[unroll]
                        for c in 0..cn {
                            // Every fragment of the slice, the bound included: the block is
                            // readable to its edge by contract, and a stale key past the
                            // prefix — NaN included — lands on a cell the mask probe
                            // overwrites with `select`, which never reads what it discards.
                            // A compare here would sit inside the unrolled walk, between
                            // every load and the mma that consumes it.
                            let mut rhs = fragment::<EI>(MatrixIdent::B, comptime!(mma.clone()));
                            rhs.copy_from(&k.at(&Region::trailing(
                                comptime!(k.space.clone()),
                                c,
                                step,
                            )));
                            accs.index_mut(comptime!(r * cn + c)).mma(
                                &lhs,
                                &rhs,
                                comptime!(Semiring::SUM_PROD),
                            );
                        }
                    }
                }
                #[unroll]
                for i in 0..comptime!(rm * cn) {
                    let (r, c) = comptime!((i / cn, i % cn));
                    let mut cell = out.at(&Region::trailing(comptime!(out.space.clone()), r, c));
                    cell.copy_from(accs.index(i));
                }
            }
            // No slice stated: every `planes`-th fragment of the whole grid, one at a time.
            None => {
                let out = self.retiled(comptime!(tiled(&self.space, m, n)));
                let k = k.retiled(comptime!(tiled(&k.space, n, kc)));
                let grid = comptime!(cols / n);
                let visits = comptime!((trailing(&self.space).0 / m) * grid);
                // Each plane takes every `planes`-th fragment, from wherever the launch put it in
                // the cube. The stride is never zero: a cube narrower than a plane cannot run a
                // plane instruction at all, and a zero step would hang rather than refuse.
                let planes = max(CUBE_DIM as usize / PLANE_DIM as usize, 1usize);
                let mut visit = UNIT_POS as usize / PLANE_DIM as usize;
                while visit < visits {
                    let row = visit / grid;
                    let col = visit % grid;
                    if col * n < cols_bound {
                        let mut acc =
                            fragment::<EA>(MatrixIdent::Accumulator, comptime!(mma.clone()));
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
        let val_dim = comptime!(trailing(&self.space).1);
        let cols = comptime!(trailing(&p.space).1);
        let steps = comptime!(cols / kc);

        self.scale_rows(factors);
        sync_cube();

        let p = p.retiled(comptime!(tiled(&p.space, m, kc)));

        match comptime!(plane_slice(&self.space)) {
            // The space states each plane's slice of the output: `{rows, val_dim / planes}`.
            // A plane holds every fragment of its slice across the whole contraction, so each
            // P fragment is loaded once and reused across the slice, and the accumulators cross
            // shared memory once per block rather than once per fragment.
            Some(slice) => {
                let planes = comptime!(val_dim / trailing(&slice).1);
                let plane = UNIT_POS as usize / PLANE_DIM as usize;
                let mine = self.at(&Region::trailing(
                    comptime!(self.space.clone()),
                    0usize,
                    plane,
                ));
                let out = mine.retiled(comptime!(tiled(&mine.space, m, n)));
                // The values this slice contracts: its own columns of every k step.
                let val = val.retiled(comptime!(tiled(&val.space, cols, val_dim / planes)));
                let val = val.at(&Region::trailing(
                    comptime!(val.space.clone()),
                    0usize,
                    plane,
                ));
                let val = val.retiled(comptime!(tiled(&val.space, kc, n)));

                let rm = comptime!(trailing(&slice).0 / m);
                let cn = comptime!(trailing(&slice).1 / n);
                let mut accs = Sequence::<Tile<EA>>::new();
                #[unroll]
                for i in 0..comptime!(rm * cn) {
                    let (r, c) = comptime!((i / cn, i % cn));
                    let mut acc = fragment::<EA>(MatrixIdent::Accumulator, comptime!(mma.clone()));
                    acc.copy_from(&out.at(&Region::trailing(comptime!(out.space.clone()), r, c)));
                    accs.push(acc);
                }
                #[unroll]
                for step in 0..steps {
                    if step * kc < cols_bound {
                        #[unroll]
                        for r in 0..rm {
                            let mut lhs = fragment::<EP>(MatrixIdent::A, comptime!(mma.clone()));
                            lhs.copy_from(&p.at(&Region::trailing(
                                comptime!(p.space.clone()),
                                r,
                                step,
                            )));
                            #[unroll]
                            for c in 0..cn {
                                let mut rhs =
                                    fragment::<EI>(MatrixIdent::B, comptime!(mma.clone()));
                                rhs.copy_from(&val.at(&Region::trailing(
                                    comptime!(val.space.clone()),
                                    step,
                                    c,
                                )));
                                accs.index_mut(comptime!(r * cn + c)).mma(
                                    &lhs,
                                    &rhs,
                                    comptime!(Semiring::SUM_PROD),
                                );
                            }
                        }
                    }
                }
                #[unroll]
                for i in 0..comptime!(rm * cn) {
                    let (r, c) = comptime!((i / cn, i % cn));
                    let mut cell = out.at(&Region::trailing(comptime!(out.space.clone()), r, c));
                    cell.copy_from(accs.index(i));
                }
            }
            // No slice stated: every `planes`-th fragment of the whole grid, one at a time.
            None => {
                let out = self.retiled(comptime!(tiled(&self.space, m, n)));
                let val = val.retiled(comptime!(tiled(&val.space, kc, n)));
                let grid = comptime!(val_dim / n);
                let visits = comptime!((trailing(&self.space).0 / m) * grid);
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
        for space in [out, lhs, rhs] {
            assert!(
                space.rank() >= 2,
                "{matmul}: a fragment leaf contracts two axes, and {space:?} has fewer"
            );
            // A rank the walk has already pinned is spanned, not iterated: a KV head cut one to a
            // cube, a batch. Anything wider would have to ride the flat index, which a fragment —
            // one origin and one row stride — cannot address.
            for i in 0..space.rank() - 2 {
                assert!(
                    space.extent_at(i) == 1,
                    "{matmul}: {space:?} carries {} positions above the two axes it contracts; \
                     window them to one, or stage the rows as a single axis",
                    space.extent_at(i)
                );
            }
        }
        let (rows, cols) = trailing(out);
        let contracted = trailing(lhs).1;
        // A visit covers the box the level the instruction sits on cuts, or the whole tile where
        // that level cut nothing. A level *above* the instruction (a plane's slice of the grid)
        // is not the fragment, so the box is read at the instruction's own level.
        let (acc_box, lhs_box) = (instruction_box(out), instruction_box(lhs));
        let (m, n) = trailing(&acc_box);
        let (lhs_rows, k) = trailing(&lhs_box);
        let stored = match layout {
            MatrixLayout::ColMajor => (cols, contracted),
            MatrixLayout::RowMajor => (contracted, cols),
            MatrixLayout::Undefined => {
                panic!("{matmul}: an operand fragment is read row- or col-major")
            }
        };
        assert!(
            trailing(lhs).0 == rows,
            "{matmul}: the lhs is {lhs:?}, not the {rows}x{contracted} this accumulator contracts"
        );
        assert!(
            trailing(rhs) == stored,
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
        let (row, col, con) = (
            out.axis_at(out.rank() - 2),
            out.axis_at(out.rank() - 1),
            lhs.axis_at(lhs.rank() - 1),
        );
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

/// `space` cut into `e0 × e1` windows on its trailing two axes, whatever it stated: the fragment
/// grid is one fact, and a leaf that reads it off the accumulator applies it to every operand. An
/// axis above those two spans its single position, so the grid over it is one window wide.
fn tiled(space: &Space, e0: usize, e1: usize) -> Space {
    let rank = space.rank();
    let extents: Vec<_> = space.axes().map(|a| (a, space.extent(a))).collect();
    let cuts: Vec<_> = extents
        .iter()
        .enumerate()
        .map(|(i, &(axis, extent))| match i {
            i if i == rank - 2 => (axis, e0),
            i if i == rank - 1 => (axis, e1),
            _ => (axis, extent),
        })
        .collect();
    Tiling::over(&mut (), &extents)
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.walk(&cuts);
        })
        .build()
}

/// The box the instruction's own level cuts, under whatever levels sit above it: a plane's
/// slice of the grid is a level, and it is not the fragment.
fn instruction_box(space: &Space) -> Space {
    let mut level = space.clone();
    while !level.is_final() && !level.divide().is_final() {
        level = level.divide();
    }
    level.sub_tile_space()
}

/// The slice one plane owns, when the space states one: a level above the instruction, cut
/// `{rows, cols / planes}`. `None` where the instruction's level is the top, which is a grid
/// every plane walks cyclically.
fn plane_slice(space: &Space) -> Option<Space> {
    if space.is_final() || space.divide().is_final() {
        return None;
    }
    let slice = space.divide();
    assert!(
        slice.divide().is_final(),
        "a fragment leaf takes at most one level above its instruction (the plane's slice); \
         {space:?} states more"
    );
    assert!(
        trailing(&slice).0 == trailing(space).0,
        "a plane's slice cuts the columns, never the rows: {space:?}"
    );
    Some(slice)
}

/// The extents of `space`'s trailing two axes — the two a matmul contracts.
fn trailing(space: &Space) -> (usize, usize) {
    let rank = space.rank();
    (space.extent_at(rank - 2), space.extent_at(rank - 1))
}
