//! The matmul reading of a [`Tile`](super::Tile): `c.mma(a, b)` treats the trailing two
//! axes as the `row × col` matrix, leading axes as a batch, and contracts `K`. A final
//! tile [`contract`](Tile::contract)s; otherwise [`mma`](Tile::mma) lowers (partition,
//! locate each operand, recurse) per the head [`Schedule`].

use cubecl::{
    cmma,
    prelude::*,
    std::tensor::{View, ViewMut, layout::Coords2d},
};

use super::*;

// The contraction is the type-aware end of the DSL: the arithmetic must know its served
// type. This is the scalar arm, served type `Vector<E, Const<1>>` (1 lane). A vectorized
// contraction (`Tile<Vector<E, N>>`, N > 1) would be a sibling impl.
#[cube]
impl<E: Numeric> Tile<Vector<E, Const<1>>> {
    /// Accumulate `lhs · rhs`, the one matmul entry point. A tile with levels left lowers
    /// per the head [`Schedule`]; a final tile [`contract`](Tile::contract)s.
    pub fn mma(&mut self, lhs: &Tile<Vector<E, Const<1>>>, rhs: &Tile<Vector<E, Const<1>>>) {
        if comptime!(self.space.is_final()) {
            self.contract(lhs, rhs);
        } else {
            match comptime!(self.space.partitioner().schedule()) {
                Schedule::Direct => mma_direct::<E>(lhs, rhs, self),
                Schedule::Staged => mma_staged::<E>(lhs, rhs, self),
                Schedule::DoubleBuffered => mma_double::<E>(lhs, rhs, self),
            }
        }
    }

    /// The [`Direct`](Schedule::Direct) lowering's per-region step.
    pub fn mma_at(
        &mut self,
        lhs: &Tile<Vector<E, Const<1>>>,
        rhs: &Tile<Vector<E, Const<1>>>,
        region: &Region,
    ) {
        self.at(region).mma(&lhs.at(region), &rhs.at(region));
    }

    /// The final contraction: `lhs · rhs` into this sub-tile. `mr × nr` are the
    /// accumulator's trailing two axes, `kc` is `lhs`'s trailing axis.
    pub fn contract(&mut self, lhs: &Tile<Vector<E, Const<1>>>, rhs: &Tile<Vector<E, Const<1>>>) {
        let (mr, nr, kc) = comptime! {
            (
                self.space.extent_at(self.space.rank() - 2),
                self.space.extent_at(self.space.rank() - 1),
                lhs.space.extent_at(lhs.space.rank() - 1)
            )
        };

        let matrices = self.matrix_count();
        for j in 0..matrices {
            let l = lhs.matrix(j);
            let r = rhs.matrix(j);
            let mut a = self.matrix_mut(j);
            mma_register::<E>(&l, &r, &mut a, mr, nr, kc);
        }
    }

    /// The cmma sibling of [`contract`](Tile::contract): `cmma::execute` accumulates
    /// `lhs · rhs` in place. [`mma`](Tile::mma) can't dispatch to it: the payload kind is
    /// a runtime enum, so the branch would emit `cmma::execute` into non-cmma backends.
    pub fn contract_cmma(
        &mut self,
        lhs: &Tile<Vector<E, Const<1>>>,
        rhs: &Tile<Vector<E, Const<1>>>,
    ) {
        match (&lhs.payload, &rhs.payload, &mut self.payload) {
            (Payload::Cmma(a), Payload::Cmma(b), Payload::Cmma(acc)) => {
                cmma::execute(&a.matrix, &b.matrix, &acc.matrix, &acc.matrix)
            }
            _ => panic!("contract_cmma: lhs, rhs, and accumulator must all be cmma fragments"),
        }
    }
}

/// All loops unroll, so the block (`c`) stays in registers: load once, run `kc`
/// rank-1 updates ([`outer_product`]), store back once.
#[cube]
fn mma_register<E: Numeric>(
    lhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    rhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    acc: &mut ViewMut<'_, Vector<E, Const<1>>, Coords2d>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
) {
    let mut c = Array::<Vector<E, Const<1>>>::new(mr * nr);
    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            c[i * nr + j] = acc.read((i as u32, j as u32).runtime());
        }
    }

    #[unroll]
    for p in 0..kc {
        outer_product::<E>(lhs, rhs, &mut c, p, mr, nr);
    }

    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            acc.write((i as u32, j as u32).runtime(), c[i * nr + j]);
        }
    }
}

/// One rank-1 update at depth `p`: the outer product of A's column and B's row,
/// accumulated into the register block `c`.
#[cube]
fn outer_product<E: Numeric>(
    lhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    rhs: &View<'_, Vector<E, Const<1>>, Coords2d>,
    c: &mut Array<Vector<E, Const<1>>>,
    #[comptime] p: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
) {
    let mut a = Array::<Vector<E, Const<1>>>::new(mr);
    let mut b = Array::<Vector<E, Const<1>>>::new(nr);

    #[unroll]
    for i in 0..mr {
        a[i] = lhs.read((i as u32, p as u32).runtime());
    }
    #[unroll]
    for j in 0..nr {
        b[j] = rhs.read((p as u32, j as u32).runtime());
    }
    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            c[i * nr + j] += a[i] * b[j];
        }
    }
}

/// `Direct`: no staging
#[cube]
pub(crate) fn mma_direct<E: Numeric>(
    lhs: &Tile<Vector<E, Const<1>>>,
    rhs: &Tile<Vector<E, Const<1>>>,
    out: &mut Tile<Vector<E, Const<1>>>,
) {
    let space = comptime!(Space::merge(&[&lhs.space, &rhs.space, &out.space]));
    let walk = Walk::over(space);
    for i in 0..walk.total() {
        out.mma_at(lhs, rhs, &walk.region(i));
    }
}

/// `Staged`: stage each operand sub-tile into shared memory, then recurse.
#[cube]
pub(crate) fn mma_staged<E: Numeric>(
    lhs: &Tile<Vector<E, Const<1>>>,
    rhs: &Tile<Vector<E, Const<1>>>,
    out: &mut Tile<Vector<E, Const<1>>>,
) {
    // The buffer's space is this level's divide, so it mirrors what `at` produces and
    // carries any remaining finer levels.
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a_smem = Shared::<[Vector<E, Const<1>>]>::new_slice(a_sub.tile_size());
    let b_smem = Shared::<[Vector<E, Const<1>>]>::new_slice(b_sub.tile_size());
    let mut a_tile = Tile::smem(&a_smem, a_sub);
    let mut b_tile = Tile::smem(&b_smem, b_sub);

    let walk = Walk::over(comptime!(Space::merge(&[
        &lhs.space, &rhs.space, &out.space
    ])));
    for i in 0..walk.total() {
        let region = walk.region(i);
        a_tile.stage(&lhs.at(&region));
        b_tile.stage(&rhs.at(&region));
        out.at(&region).mma(&a_tile, &b_tile);
    }
}

/// `DoubleBuffered`: two staged buffers per operand, prefetching the next region
/// into the idle slot while computing the current one. The slot choice is comptime
/// because the two ping-pong phases are written out explicitly.
#[cube]
pub(crate) fn mma_double<E: Numeric>(
    lhs: &Tile<Vector<E, Const<1>>>,
    rhs: &Tile<Vector<E, Const<1>>>,
    out: &mut Tile<Vector<E, Const<1>>>,
) {
    // Allocated here in caller scope because a view-backed buffer must outlive the ring.
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a0 = Shared::<[Vector<E, Const<1>>]>::new_slice(a_sub.tile_size());
    let a1 = Shared::<[Vector<E, Const<1>>]>::new_slice(a_sub.tile_size());
    let b0 = Shared::<[Vector<E, Const<1>>]>::new_slice(b_sub.tile_size());
    let b1 = Shared::<[Vector<E, Const<1>>]>::new_slice(b_sub.tile_size());
    let mut a_buf = Sequence::new();
    a_buf.push(Tile::smem(&a0, comptime!(a_sub.clone())));
    a_buf.push(Tile::smem(&a1, comptime!(a_sub.clone())));
    let mut b_buf = Sequence::new();
    b_buf.push(Tile::smem(&b0, comptime!(b_sub.clone())));
    b_buf.push(Tile::smem(&b1, comptime!(b_sub.clone())));
    let mut a = Ring::new(a_buf);
    let mut b = Ring::new(b_buf);

    let walk = Walk::over(comptime!(Space::merge(&[
        &lhs.space, &rhs.space, &out.space
    ])));

    // prologue: prime slot 0 with region 0.
    let r0 = walk.region(0);
    a.stage(0usize, &lhs.at(&r0));
    b.stage(0usize, &rhs.at(&r0));
    sync_cube();

    let n = walk.total();
    for p in 0..n / 2 {
        let even = p * 2;
        let odd = even + 1;

        // phase 0: prefetch the odd region into slot 1, compute the even region.
        a.stage(1usize, &lhs.at(&walk.region(even + 1)));
        b.stage(1usize, &rhs.at(&walk.region(even + 1)));
        out.at(&walk.region(even)).mma(a.get(0usize), b.get(0usize));
        sync_cube();

        // phase 1: prefetch the next even region into slot 0, compute the odd region.
        if odd + 1 < n {
            a.stage(0usize, &lhs.at(&walk.region(odd + 1)));
            b.stage(0usize, &rhs.at(&walk.region(odd + 1)));
        }
        out.at(&walk.region(odd)).mma(a.get(1usize), b.get(1usize));
        sync_cube();
    }
}
