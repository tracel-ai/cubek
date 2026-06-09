//! The matmul reading of a [`Tile`](super::Tile): `c.mma(a, b)` treats the trailing two
//! axes as the `row × col` matrix, leading axes as a batch, and contracts `K`. A final
//! tile contracts via the [`Mma`] trait; otherwise [`mma`](Tile::mma) lowers (partition,
//! locate each operand, recurse) per the head [`Schedule`].

use cubecl::{
    prelude::*,
    std::tensor::{View, ViewMut, layout::Coords2d},
};

use super::*;

/// The leaf contraction `acc += lhs · rhs`, reached only at a final tile. Keyed on the
/// accumulator's element so the generic lowering can name the bound; the method takes the whole
/// tile, so it already has the acc's space. The impls that exist are the legal patterns.
#[cube]
pub trait Mma<Lhs: CubePrimitive, Rhs: CubePrimitive>: CubePrimitive {
    fn mma(acc: &mut Tile<Self>, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>);
}

/// Same-`E` operands: `rhs`/`acc` line `N` by `V` (the SIMD load), `lhs` lines `K` by `L`
/// (broadcast); `V = L = Const<1>` is the scalar kernel. The accumulator's storage picks the
/// path: cmma fragments via `cmma::execute`, memory tiles via the register microkernel.
#[cube]
impl<E: Numeric, V: Size, L: Size> Mma<Vector<E, L>, Vector<E, V>> for Vector<E, V> {
    fn mma(acc: &mut Tile<Vector<E, V>>, lhs: &Tile<Vector<E, L>>, rhs: &Tile<Vector<E, V>>) {
        let space = comptime!(acc.space.clone());
        let payload = &mut acc.payload;
        match payload {
            Payload::Cmma(d) => d.mma(lhs, rhs),
            Payload::Gmem(g) | Payload::Smem(g) => mma_register_memory::<E, L, V>(g, lhs, rhs, space),
        }
    }
}

#[cube]
impl<Acc: CubePrimitive> Tile<Acc> {
    /// `c.mma(a, b)`: a tile with levels left lowers per its [`Schedule`]; a final tile
    /// contracts via [`Mma`].
    pub fn mma<Lhs: CubePrimitive, Rhs: CubePrimitive>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>)
    where
        Acc: Mma<Lhs, Rhs>,
    {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final => Acc::mma(self, lhs, rhs),
            Partitioner::Level(level) => match level.schedule() {
                Schedule::Direct => mma_direct(lhs, rhs, self),
                Schedule::Staged => mma_staged(lhs, rhs, self),
                Schedule::DoubleBuffered => mma_double(lhs, rhs, self),
            },
        }
    }

    /// The [`Direct`](Schedule::Direct) lowering's per-region step.
    pub fn mma_at<Lhs: CubePrimitive, Rhs: CubePrimitive>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        region: &Region,
    ) where
        Acc: Mma<Lhs, Rhs>,
    {
        self.at(region).mma(&lhs.at(region), &rhs.at(region));
    }
}

/// Run the register microkernel over each batch matrix. `mr × nr` are the accumulator's
/// trailing axes (`nr` in `N`-lines); `kc` is scalar `K`, read off `rhs` (whose `K` is unlined).
#[cube]
pub(crate) fn mma_register_memory<E: Numeric, L: Size, V: Size>(
    acc: &mut MemData<Vector<E, V>>,
    lhs: &Tile<Vector<E, L>>,
    rhs: &Tile<Vector<E, V>>,
    #[comptime] space: Space,
) {
    let (mr, nr, kc) = comptime! {
        (
            space.extent_at(space.rank() - 2),
            space.extent_at(space.rank() - 1),
            rhs.space.extent_at(rhs.space.rank() - 2)
        )
    };

    let matrices = comptime! {
        let mut count = 1;
        for p in 0..space.rank() - 2 {
            count *= space.extent_at(p);
        }
        count
    };

    for j in 0..matrices {
        let l = lhs.matrix(j);
        let r = rhs.matrix(j);
        let mut a = acc.matrix_mut(j, comptime!(space.clone()));
        mma_register::<E, L, V>(&l, &r, &mut a, mr, nr, kc);
    }
}

/// The microkernel. The `mr × nr` block of `V`-wide accumulators lives in registers: load once,
/// run `kc` rank-1 updates ([`outer_product`]), store once. `nr` counts `N`-lines.
#[cube]
fn mma_register<E: Numeric, L: Size, V: Size>(
    lhs: &View<'_, Vector<E, L>, Coords2d>,
    rhs: &View<'_, Vector<E, V>, Coords2d>,
    acc: &mut ViewMut<'_, Vector<E, V>, Coords2d>,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
    #[comptime] kc: usize,
) {
    let mut c = Array::<Vector<E, V>>::new(mr * nr);
    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            c[i * nr + j] = acc.read((i as u32, j as u32).runtime());
        }
    }

    #[unroll]
    for p in 0..kc {
        outer_product::<E, L, V>(lhs, rhs, &mut c, p, mr, nr);
    }

    #[unroll]
    for i in 0..mr {
        #[unroll]
        for j in 0..nr {
            acc.write((i as u32, j as u32).runtime(), c[i * nr + j]);
        }
    }
}

/// One rank-1 update at scalar depth `p`: `c += outer(A[:, p], B[p, :])`. `A[i, p]` is lane
/// `p % L` of `lhs`'s `(p / L)` `K`-line, broadcast and multiplied by `B`'s `V`-wide lines.
#[cube]
fn outer_product<E: Numeric, L: Size, V: Size>(
    lhs: &View<'_, Vector<E, L>, Coords2d>,
    rhs: &View<'_, Vector<E, V>, Coords2d>,
    c: &mut Array<Vector<E, V>>,
    #[comptime] p: usize,
    #[comptime] mr: usize,
    #[comptime] nr: usize,
) {
    let l = comptime!(L::value());
    let mut b = Array::<Vector<E, V>>::new(nr);
    #[unroll]
    for j in 0..nr {
        b[j] = rhs.read((p as u32, j as u32).runtime());
    }
    #[unroll]
    for i in 0..mr {
        let scalar = lhs
            .read((i as u32, (p / l) as u32).runtime())
            .extract(p % l);
        let a = Vector::<E, V>::cast_from(scalar);
        #[unroll]
        for j in 0..nr {
            c[i * nr + j] += a * b[j];
        }
    }
}

/// `Direct`: no staging
#[cube]
pub(crate) fn mma_direct<Lhs: CubePrimitive, Rhs: CubePrimitive, Acc>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
) where
    Acc: CubePrimitive + Mma<Lhs, Rhs>,
{
    let space = comptime!(Space::merge(&[&lhs.space, &rhs.space, &out.space]));
    let walk = Walk::over(space);
    for i in 0..walk.total() {
        out.mma_at(lhs, rhs, &walk.region(i));
    }
}

/// `Staged`: stage each operand sub-tile into shared memory, then recurse. Each buffer keeps
/// its own served type.
#[cube]
pub(crate) fn mma_staged<Lhs: CubePrimitive, Rhs: CubePrimitive, Acc>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
) where
    Acc: CubePrimitive + Mma<Lhs, Rhs>,
{
    // The buffer's space is this level's divide, so it mirrors what `at` produces and
    // carries any remaining finer levels.
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a_smem = Shared::<[Lhs]>::new_slice(a_sub.tile_size());
    let b_smem = Shared::<[Rhs]>::new_slice(b_sub.tile_size());
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

/// `DoubleBuffered`: two staged buffers per operand, prefetching the next region into the idle
/// slot while computing the current one.
#[cube]
pub(crate) fn mma_double<Lhs: CubePrimitive, Rhs: CubePrimitive, Acc>(
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    out: &mut Tile<Acc>,
) where
    Acc: CubePrimitive + Mma<Lhs, Rhs>,
{
    // Allocated here in caller scope because a view-backed buffer must outlive the ring.
    let a_sub = comptime!(lhs.space.divide());
    let b_sub = comptime!(rhs.space.divide());
    let a0 = Shared::<[Lhs]>::new_slice(a_sub.tile_size());
    let a1 = Shared::<[Lhs]>::new_slice(a_sub.tile_size());
    let b0 = Shared::<[Rhs]>::new_slice(b_sub.tile_size());
    let b1 = Shared::<[Rhs]>::new_slice(b_sub.tile_size());
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
