//! The tensor-core encoding of a plane tile ([`CmmaData`]) and its fragment↔memory transports.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// A tensor-core fragment plus the comptime config its load/store paths dispatch on.
/// `Clone` duplicates the handle, not the fragment.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct CmmaData<T: Numeric> {
    pub matrix: Matrix<T>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    /// The whole MMA tile's `(m, n)`, whatever the role.
    #[cube(comptime)]
    pub shape: (usize, usize),
    /// This plane's one-tile window of shared memory the fragment bounces through.
    pub scratch: ComptimeOption<Shared<[T]>>,
}

#[cube]
impl<T: Numeric> CmmaData<T> {
    /// Allocate an uninitialized fragment. `m`/`n`/`k` are the whole MMA tile, whatever the role.
    pub(crate) fn alloc(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
    ) -> CmmaData<T> {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, layout) };
        CmmaData::<T> {
            matrix,
            ident,
            layout,
            shape: comptime!((m, n)),
            scratch: ComptimeOption::new_None(),
        }
    }

    /// This fragment carrying `scratch`, one tile of shared memory for its plane.
    pub(crate) fn with_scratch(self, scratch: Shared<[T]>) -> CmmaData<T> {
        CmmaData::<T> {
            matrix: self.matrix,
            ident: comptime!(self.ident),
            layout: comptime!(self.layout),
            shape: comptime!(self.shape),
            scratch: ComptimeOption::new_Some(scratch),
        }
    }

    /// Store this tile row-major into `scratch`, one tile's cells from its start.
    pub(crate) fn store_scratch(&self, scratch: &Shared<[T]>) {
        let n = comptime!(self.shape.1 as u32);
        let mut window = scratch.clone();
        cmma::store(&mut window, &self.matrix, n, MatrixLayout::RowMajor)
    }

    /// Load this tile back from `scratch`.
    pub(crate) fn load_scratch(&mut self, scratch: &Shared<[T]>) {
        let n = comptime!(self.shape.1 as u32);
        cmma::load_with_layout(&mut self.matrix, scratch, n, MatrixLayout::RowMajor)
    }

    /// Zero the fragment.
    pub(crate) fn zero(&mut self) {
        cmma::fill(&mut self.matrix, T::from_int(0));
    }

    /// Fill this fragment from `mem`'s window, stepping rows by the physical row stride.
    pub(crate) fn load_window(&mut self, mem: &Memory<T>, #[comptime] row: usize) {
        let element = mem.stage_element();
        comptime!(assert!(
            element == StageElement::Served,
            "CmmaData::load_window: a cmma fragment loads at one element type, so it cannot \
             unpack a packed source as it reads; land it first (`landed_for`) or decode it into a \
             stage (`stage.copy_from(&w.mul(&scales))`)"
        ));
        let stride = mem.row_stride_at(row);
        match comptime!(self.ident) {
            MatrixIdent::Accumulator => cmma::load_with_layout(
                &mut self.matrix,
                mem.window_slice(),
                stride,
                comptime!(self.layout),
            ),
            _ => cmma::load(&mut self.matrix, mem.window_slice(), stride),
        }
    }

    /// Drain this fragment into `mem`'s *window* (origin offset, physical row stride).
    pub(crate) fn store_window(&self, mem: &mut Memory<T>, #[comptime] row: usize) {
        let stride = mem.row_stride_at(row);
        cmma::store(
            mem.window_slice_mut(),
            &self.matrix,
            stride,
            comptime!(self.layout),
        )
    }

    /// `self[i, :] *= factors[first + i]` for every slice `i` along the fragment's columns,
    /// bounced through its slot of the plane's scratch: spilled, scaled by the plane's units in
    /// turns, and reloaded, each step met on `sync_plane`. The slot is the plane's own, so no
    /// other plane waits; every unit of the plane calls it.
    pub(crate) fn mul_along(&self, factors: &Array<T>, #[comptime] first: usize) {
        self.spill_to_scratch();
        sync_plane();
        self.mul_spilled(factors, first);
        sync_plane();
        self.reload_from_scratch();
        sync_plane();
    }

    /// `slot[i, :] *= factors[first + i]` over this fragment's spilled cells, the plane's units
    /// taking them in turns. The caller owns the barriers.
    pub(crate) fn mul_spilled(&self, factors: &Array<T>, #[comptime] first: usize) {
        let mut scratch = self.scratch_slot();
        let (m, n) = self.shape;
        let mut cell = UNIT_POS_PLANE as usize;
        while cell < m * n {
            scratch[cell] *= factors[first + cell / n];
            cell += PLANE_DIM as usize;
        }
    }

    /// Load this fragment back from its slot of the plane's scratch. The caller owns the barriers.
    pub(crate) fn reload_from_scratch(&self) {
        let scratch = self.scratch_slot();
        let mut fragment = self.clone();
        fragment.load_scratch(&scratch);
    }

    /// Store this fragment into its slot of the plane's scratch. The caller owns the barriers.
    pub(crate) fn spill_to_scratch(&self) {
        self.store_scratch(&self.scratch_slot());
    }

    /// Write this fragment's spilled cells into `mem` through the store's own write.
    pub(crate) fn add_from_scratch<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        let scratch = self.scratch_slot();
        let (m, n) = comptime!(self.shape);
        let width = comptime!(mem.store.vector_size);
        comptime!(assert!(
            n.is_multiple_of(width),
            "CmmaData::add_from_scratch: the store's lines ({width}) do not divide the \
             fragment's columns ({n})"
        ));
        let size!(W) = width;
        let lines_per_row = comptime!(n / width);
        let lines = comptime!(m * lines_per_row);
        let axes = comptime!(MatrixAxes::trailing(&space));
        let mut sink = mem.matrix_mut::<W>(0usize, axes, space);
        let plane_units = PLANE_DIM as usize;
        let mut line = UNIT_POS_PLANE as usize;
        while line < lines {
            let mut value = Vector::<Out, W>::empty();
            #[unroll]
            for e in 0..width {
                value.insert(e, Out::cast_from(scratch[line * width + e]));
            }
            sink.write(
                ((line / lines_per_row) as u32, (line % lines_per_row) as u32),
                value,
            );
            line += plane_units;
        }
    }

    /// Drain this fragment through the plane's scratch, with cube-wide syncs.
    pub(crate) fn bounce_cast_window<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        sync_cube();
        self.spill_to_scratch();
        sync_cube();
        self.add_from_scratch(mem, space);
        sync_cube();
    }

    /// This fragment's slot of the plane's scratch, or the reason there is none.
    fn scratch_slot(&self) -> Shared<[T]> {
        #[comptime]
        match &self.scratch {
            ComptimeOption::Some(scratch) => scratch.clone(),
            ComptimeOption::None => panic!(
                "CmmaData: a fragment bounces through a scratch, to scale its slices, to reach a \
                 store that folds, or a window the problem's edge cuts short; open the \
                 accumulator with `with_scratch`"
            ),
        }
    }

    /// Drain this fragment into `mem`'s window, casting `T` down to the sink's element type.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] row: usize,
    ) {
        let stride = mem.row_stride_at(row);
        let casted: Matrix<Out> = cmma::cast(&self.matrix);
        cmma::store(
            mem.window_slice_mut(),
            &casted,
            stride,
            comptime!(self.layout),
        )
    }
}

/// How a cmma fragment reaches its destination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FragmentDrain {
    /// The fragment's store intrinsic: a replacing destination, window wholly inside it.
    Intrinsic,
    /// Through the plane's scratch: a destination that adds, or a window cut short by the edge.
    Bounce,
}

impl FragmentDrain {
    /// How a fragment drains into a destination written as `access` says, `addressed` when its
    /// values sit at an address the intrinsic can store to.
    pub(crate) const fn of(access: &Access, addressed: bool) -> Self {
        match (access.write, access.overhang, addressed) {
            (Write::Replace, Overhang::Never | Overhang::Fits, true) => FragmentDrain::Intrinsic,
            (Write::Replace, _, _) | (Write::Accumulate | Write::Relay, _, _) => {
                FragmentDrain::Bounce
            }
        }
    }
}

#[cfg(test)]
mod fragment_drain_tests {
    use super::*;

    fn access(write: Write, overhang: Overhang) -> Access {
        Access {
            whole: false,
            overhang,
            write,
            fill: FillUnits::cube(0),
            storage: WindowStorage::Stored(Storage::Strided),
            delivery: Delivery::SyncPerUnit,
        }
    }

    #[test]
    fn a_replacing_window_inside_its_buffer_stores_through_the_intrinsic() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            let drain = FragmentDrain::of(&access(Write::Replace, overhang), true);
            assert_eq!(drain, FragmentDrain::Intrinsic);
        }
    }

    #[test]
    fn an_overhanging_or_folding_window_bounces() {
        let masked = FragmentDrain::of(&access(Write::Replace, Overhang::Masked), true);
        assert_eq!(masked, FragmentDrain::Bounce);
        for overhang in [Overhang::Never, Overhang::Fits, Overhang::Masked] {
            for write in [Write::Accumulate, Write::Relay] {
                assert_eq!(
                    FragmentDrain::of(&access(write, overhang), true),
                    FragmentDrain::Bounce,
                    "{write:?} {overhang:?}"
                );
            }
        }
    }

    /// A write call has no address for the intrinsic to store to, so even a replacing window
    /// wholly inside it bounces.
    #[test]
    fn a_window_with_no_address_bounces() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            let drain = FragmentDrain::of(&access(Write::Replace, overhang), false);
            assert_eq!(drain, FragmentDrain::Bounce);
        }
    }
}
