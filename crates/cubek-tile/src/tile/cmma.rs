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
    /// where its intrinsic cannot do the work: a row-wise op, or a drain into a store that folds.
    /// Opened on the accumulator ([`with_scratch`](Tile::with_scratch)) and carried by every
    /// fragment taken off it.
    pub scratch: ComptimeOption<Shared<[T]>>,
    /// Units in one plane, stated with the scratch: what a bounce deals the tile's cells across.
    #[cube(comptime)]
    pub lanes: usize,
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
            lanes: 0usize,
        }
    }

    /// This fragment carrying `scratch`, one tile of shared memory for its plane of `lanes`.
    pub(crate) fn with_scratch(
        self,
        scratch: Shared<[T]>,
        #[comptime] lanes: usize,
    ) -> CmmaData<T> {
        CmmaData::<T> {
            matrix: self.matrix,
            ident: comptime!(self.ident),
            layout: comptime!(self.layout),
            shape: comptime!(self.shape),
            scratch: ComptimeOption::new_Some(scratch),
            lanes,
        }
    }

    /// The whole MMA tile's `(m, n)`.
    pub(crate) fn shape(&self) -> comptime_type!((usize, usize)) {
        comptime!(self.shape)
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

    /// An uninitialized fragment presented as a `Cmma` tile. `m`/`n`/`k` are the whole
    /// MMA tile, passed in full whatever the role.
    pub fn fragment(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] space: Space,
    ) -> Tile<T> {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, layout) };
        Tile::<T> {
            tile_kind: TileKind::new_PlaneTile(PlaneTile::new_Cmma(CmmaData::<T> {
                matrix,
                ident,
                layout,
                shape: comptime!((m, n)),
                scratch: ComptimeOption::new_None(),
                lanes: 0usize,
            })),
            space: comptime!(space),
            depth: comptime!(0usize),
            levels: comptime!(Vec::new()),
        }
    }

    /// Zero the fragment.
    pub(crate) fn zero(&mut self) {
        cmma::fill(&mut self.matrix, T::from_int(0));
    }

    /// Fill this fragment from `mem`'s *window*: `A`/`B` use `cmma::load`, an
    /// `Accumulator` uses `load_with_layout`. Rows step by the store's physical row
    /// stride, so a window into a larger stage loads like a whole buffer.
    pub(crate) fn load_window(&mut self, mem: &MemData<T>) {
        let dequant_at = mem.dequant_at();
        comptime!(assert!(
            dequant_at == DequantAt::Load,
            "CmmaData::load_window: a cmma fragment loads at one element type, so it cannot \
             decode a quantized source as it reads; serve that operand by its load \
             (DequantAt::Load) or stage it into shared memory first"
        ));
        let stride = mem.row_stride();
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
    pub(crate) fn store_window(&self, mem: &mut MemData<T>) {
        let stride = mem.row_stride();
        cmma::store(
            mem.window_slice_mut(),
            &self.matrix,
            stride,
            comptime!(self.layout),
        )
    }

    /// Drain this fragment into a store that folds: bounced through the plane's scratch, then
    /// each lane adds its cells through the store's own write, which is the atomic add. The
    /// intrinsic's store replaces and elects no writer; the scratch is what gives each cell one
    /// owner, so a lane adds it once. The syncs are cube-wide, as every fragment bounce's are.
    pub(crate) fn accumulate_cast_window<Out: Numeric>(
        &self,
        mem: &mut MemData<Out>,
        #[comptime] space: Space,
    ) {
        let scratch = #[comptime]
        match &self.scratch {
            ComptimeOption::Some(scratch) => scratch.clone(),
            ComptimeOption::None => panic!(
                "CmmaData::accumulate_cast_window: a fragment folds into an accumulating store \
                 through a scratch; open the accumulator with `with_scratch`"
            ),
        };
        let (m, n) = comptime!(self.shape);
        // The store takes lines of its own width, so the tile is dealt out in lines: `n` is the
        // instruction's and a served width divides it.
        let width = comptime!(mem.store.vector_size);
        comptime!(assert!(
            n.is_multiple_of(width),
            "CmmaData::accumulate_cast_window: the store's lines ({width}) do not divide the \
             fragment's columns ({n})"
        ));
        let size!(W) = width;
        let lines_per_row = comptime!(n / width);
        let lines = comptime!(m * lines_per_row);
        let lanes = comptime!(self.lanes);
        let lane = UNIT_POS_X as usize % lanes;
        let axes = comptime!(MatrixAxes::trailing_pair(&space));
        let mut sink = mem.matrix_mut::<W>(0usize, axes, space);
        sync_cube();
        self.store_scratch(&scratch);
        sync_cube();
        #[unroll]
        for t in 0..comptime!(lines.div_ceil(lanes)) {
            let line = lane + t * lanes;
            if comptime!(lines.is_multiple_of(lanes)) || line < lines {
                let mut value = Vector::<Out, W>::empty();
                #[unroll]
                for e in 0..width {
                    value.insert(e, Out::cast_from(scratch[line * width + e]));
                }
                sink.write(
                    ((line / lines_per_row) as u32, (line % lines_per_row) as u32),
                    value,
                );
            }
        }
        sync_cube();
    }

    /// Drain this fragment into `mem`'s *window*, casting `T` down to the sink's element
    /// type first: a register accumulator (e.g. `f32`) is wider than the stored output
    /// (e.g. `f16`). The cast is a no-op when the types match.
    pub(crate) fn store_cast_window<Out: Numeric>(&self, mem: &mut MemData<Out>) {
        let stride = mem.row_stride();
        let casted: Matrix<Out> = cmma::cast(&self.matrix);
        cmma::store(
            mem.window_slice_mut(),
            &casted,
            stride,
            comptime!(self.layout),
        )
    }
}
