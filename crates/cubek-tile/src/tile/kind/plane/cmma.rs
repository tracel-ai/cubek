//! The tensor-core encoding of a plane tile ([`CmmaData`]) and its fragment↔memory transports.
//! The grid over it is the encoding-blind [`PlanePartition`](super::plane).

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// A tensor-core fragment plus the comptime config its load/store paths dispatch on.
/// `Clone` duplicates the handle, not the fragment: a clone is the same matrix.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CmmaData<T: Numeric> {
    pub matrix: Matrix<T>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    /// The whole MMA tile's `(m, n)`, whatever the role.
    #[cube(comptime)]
    pub shape: (usize, usize),
    /// This plane's window of shared memory, one tile wide, that the fragment bounces through
    /// where its intrinsic cannot do the work (a row-wise op, a drain into a folding store). Opened
    /// on the accumulator ([`with_scratch`](Tile::with_scratch)) and carried by every fragment.
    pub scratch: ComptimeOption<Shared<[T]>>,
}

#[cube]
impl<T: Numeric> CmmaData<T> {
    /// Allocate an uninitialized fragment. `m`/`n`/`k` are the whole MMA tile, passed in
    /// full whatever the role; `layout` is how the stage it loads from lays the role's rows
    /// out.
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

    /// An uninitialized fragment presented as a `Cmma` tile, cut by no level: the form a test
    /// hands the leaf straight. `m`/`n`/`k` are the whole MMA tile, passed in full whatever the
    /// role.
    pub fn fragment(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] space: Space,
    ) -> Tile<T> {
        Tile::<T> {
            kind: TileKind::new_PlaneTile(PlaneTile::new_Cmma(CmmaData::<T>::alloc(
                ident, m, n, k, layout,
            ))),
            place: comptime!(Placement::new(space, 0usize, Vec::new())),
        }
    }

    /// Zero the fragment.
    pub(crate) fn zero(&mut self) {
        cmma::fill(&mut self.matrix, T::from_int(0));
    }

    /// Fill this fragment from `mem`'s *window*: `A`/`B` use `cmma::load`, an
    /// `Accumulator` uses `load_with_layout`. Rows step by the store's physical row
    /// stride, so a window into a larger stage loads like a whole buffer.
    pub(crate) fn load_window(&mut self, mem: &Memory<T>, #[comptime] row: usize) {
        let dequant_at = mem.dequant_at();
        comptime!(assert!(
            dequant_at == DequantAt::Load,
            "CmmaData::load_window: a cmma fragment loads at one element type, so it cannot \
             decode a quantized source as it reads; serve that operand by its load \
             (DequantAt::Load) or stage it into shared memory first"
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

    /// Store this fragment into its own slot of the plane's scratch: the first half of a bounce,
    /// and the one thing a fragment can do with its cells.
    ///
    /// **The caller owns the barriers.** A whole partition spilled together pays them once for the
    /// drain instead of once per tile, which is the whole point of the scratch being sizeable.
    pub(crate) fn spill_to_scratch(&self) {
        self.store_scratch(&self.scratch_slot("spill_to_scratch"));
    }

    /// Add this fragment's spilled cells into `mem` through the store's own write, which for a
    /// folding store is the atomic add: the second half of a bounce. The intrinsic's store
    /// replaces and elects no writer; the scratch is what gives each cell one owner.
    ///
    /// Lines of the store's width rather than scalars, and the lanes deal them between
    /// themselves, so every cell has exactly one owner and lands once.
    pub(crate) fn add_from_scratch<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        let scratch = self.scratch_slot("add_from_scratch");
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
        // The plane's own width, which the hardware states, so the lanes deal the lines
        // between them whatever shape the launch gave the cube.
        let lanes = PLANE_DIM as usize;
        let mut line = UNIT_POS_X as usize;
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
            line += lanes;
        }
    }

    /// Drain this fragment into a store that folds, on its own: spill, wait, add, wait. What a
    /// partition drained a tile at a time runs, and the barriers a whole-partition drain hoists.
    pub(crate) fn accumulate_cast_window<Out: Numeric>(
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
    fn scratch_slot(&self, #[comptime] site: &str) -> Shared<[T]> {
        #[comptime]
        match &self.scratch {
            ComptimeOption::Some(scratch) => scratch.clone(),
            ComptimeOption::None => panic!(
                "CmmaData::{site}: a fragment folds into an accumulating store through a \
                 scratch; open the accumulator with `with_scratch`"
            ),
        }
    }

    /// Drain this fragment into `mem`'s *window*, casting `T` down to the sink's element
    /// type first: a register accumulator (e.g. `f32`) is wider than the stored output
    /// (e.g. `f16`). The cast is a no-op when the types match.
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
