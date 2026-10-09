//! The warpgroup MMA encoding of a plane group's tile ([`WgmmaData`]): the `64 × n` accumulator
//! the group's four planes write together, its MMAs reading both operands straight out of shared
//! memory through matrix descriptors.

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;
use cubecl::wgmma::{
    Accumulator, Major, MatrixDescriptor, Swizzle, WARPGROUP_M, WARPGROUP_UNITS, WgmmaTileLayout,
};

use crate::ops::matmul::leaf::window_layouts;
use crate::*;

/// A plane group's accumulator, handed to the MMAs that write it until the drain waits for them.
/// `Clone` duplicates the handle, not the registers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct WgmmaData<T: Numeric> {
    pub(crate) acc: Pending<Accumulator<T>>,
    #[cube(comptime)]
    pub n: usize,
}

#[cube]
impl<T: Numeric> WgmmaData<T> {
    /// A zeroed `m × n` accumulator, `m` the warpgroup's 64 rows.
    pub(crate) fn new(#[comptime] m: usize, #[comptime] n: usize) -> WgmmaData<T> {
        comptime!(assert!(
            m == WARPGROUP_M,
            "WgmmaData: a warpgroup MMA computes {WARPGROUP_M} rows, and this tile has {m}; give \
             each plane group a {WARPGROUP_M}-row tile"
        ));
        WgmmaData::<T> {
            acc: Accumulator::<T>::new(n).start(),
            n,
        }
    }

    /// Zero the accumulator: a fresh one, handed to the MMAs to come.
    pub(crate) fn zero(&mut self) {
        self.acc = Accumulator::<T>::new(comptime!(self.n)).start();
    }

    /// Issue `self += lhs · rhs` over the whole of the windows' `k`, one MMA a step. Both windows
    /// lie in shared memory, K-major, their rows one swizzle span long.
    pub(crate) fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &Tile<L>, rhs: &Tile<R>) {
        let (lhs_layout, rhs_layout) =
            comptime!(window_layouts(&lhs.place.space, &rhs.place.space));
        comptime!(assert!(
            lhs_layout == MatrixLayout::RowMajor && rhs_layout == MatrixLayout::ColMajor,
            "WgmmaData::mma: a warpgroup MMA reads both operands K-major here, the lhs `{{m, k}}` \
             and the rhs `{{n, k}}`; stage them so"
        ));
        let k = comptime!(trailing_extent(&lhs.place.space));
        let elem_size = L::size().comptime();
        let k_step = comptime!(WgmmaTileLayout::k_step(elem_size));
        comptime!(assert!(
            k.is_multiple_of(k_step),
            "WgmmaData::mma: a window of {k} along k is not a whole number of the MMA's {k_step}"
        ));
        let a = descriptor::<L>(lhs, WARPGROUP_M, k);
        let b = descriptor::<R>(rhs, comptime!(self.n), k);
        #[unroll]
        for step in 0..comptime!(k / k_step) {
            let at = step * k_step;
            self.acc.execute(&a.at(0usize, at), &b.at(0usize, at));
        }
    }

    /// Commit the MMAs issued since the last commit, and return their completion: once it
    /// resolves, they are done reading their stage.
    pub(crate) fn commit(&mut self) -> Pending<()> {
        self.acc.commit()
    }

    /// Wait for every MMA, then drain the accumulator into `mem`'s window, cast to its element:
    /// each unit writes the cells it holds.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        comptime!(assert!(
            mem.store.vector_size == 1,
            "WgmmaData: an accumulator drained cell by cell writes one element at a time, and \
             its destination is bound {} wide; bind it one element wide",
            mem.store.vector_size
        ));
        let acc = self.acc.clone().wait();
        // The groups start at plane 0, so a unit's place in its group is its place in the cube
        // modulo the group.
        let unit = UNIT_POS % comptime!(WARPGROUP_UNITS as u32);
        let axes = comptime!(MatrixAxes::trailing(&space));
        let mut sink = mem.matrix_mut::<Const<1>>(0usize, axes, space);
        #[unroll]
        for nth in 0..acc.len() {
            let at = acc.position_of_nth(unit, nth as u32);
            let mut value = Vector::<Out, Const<1>>::empty();
            value.insert(0usize, Out::cast_from(acc.get(nth)));
            sink.write(at, value);
        }
    }
}

/// The extent of `space`'s trailing axis, the contiguous one.
fn trailing_extent(space: &Space) -> usize {
    space.extent_at(space.rank() - 1)
}

/// The descriptor of `tile`'s `rows × k` window, which starts a block of the swizzle's rows: its
/// first line sits where the stage's swizzle leaves it. The tile it describes is the stage's,
/// whose rows are one swizzle span long whatever of them the window holds: what the MMA steps
/// from the window's origin.
#[cube]
fn descriptor<E: Numeric>(
    tile: &Tile<E>,
    #[comptime] rows: usize,
    #[comptime] k: usize,
) -> MatrixDescriptor<E> {
    let elem_size = E::size().comptime();
    let mem = tile.mem("wgmma");
    let swizzle = comptime!(stage_swizzle(&mem.layout.rows));
    let row = comptime!(swizzle.width() / elem_size);
    comptime!(assert!(
        k <= row,
        "WgmmaData: a window of {k} along k runs past its stage's row of {row}"
    ));
    let layout = comptime!(WgmmaTileLayout {
        major: Major::K,
        swizzle,
        rows,
        k: row,
    });
    let size!(W) = tile.vector_size();
    let view = tile.fragment_matrix_packed::<W>(rows, k);
    let origin = view.line_slice(
        (0u32.runtime(), 0u32.runtime()),
        (1u32.runtime(), 1u32.runtime()),
    );
    MatrixDescriptor::<E>::new(origin, layout)
}

/// The swizzle a stage keeps its rows in, which a warpgroup MMA reads them under: a swizzled row
/// a TMA descriptor lands is one swizzle span long, which lies as the MMA's tile does.
fn stage_swizzle(rows: &RowArrangement) -> Swizzle {
    match rows.tma_swizzle() {
        Ok(TensorMapSwizzle::B32) => Swizzle::B32,
        Ok(TensorMapSwizzle::B64) => Swizzle::B64,
        Ok(TensorMapSwizzle::B128) => Swizzle::B128,
        Ok(other) => panic!(
            "WgmmaData: the stage keeps its rows under {other:?}, which no warpgroup MMA reads; \
             stage them RowChunks::Swizzled, one swizzle span a row"
        ),
        Err(why) => panic!("WgmmaData: a warpgroup MMA cannot read the stage: {why}"),
    }
}
