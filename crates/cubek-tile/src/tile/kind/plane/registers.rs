//! The register-resident accumulator ([`RegisterData`]), the software leaf's counterpart to
//! a cmma fragment.

use cubecl::prelude::*;

use crate::*;

// The block's line width, bound by `alloc` via `register_size`. Allocated at the vector
// element: the CPU backend refuses a scalar array re-viewed as lines.
define_size!(pub(crate) RA);

/// What a register block accumulates: products of two operands under a semiring, or one
/// operand's values under a monoid.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Accumulation {
    Contraction(Semiring),
    Reduction(Monoid),
}

impl Accumulation {
    /// The `⊕` partials merge under, and the identity a block opens at.
    pub(crate) fn monoid(self) -> Monoid {
        match self {
            Accumulation::Contraction(semiring) => semiring.add(),
            Accumulation::Reduction(monoid) => monoid,
        }
    }

    /// The semiring a contraction runs under; a reduction runs none.
    pub(crate) fn semiring(self, site: &str) -> Semiring {
        match self {
            Accumulation::Contraction(semiring) => semiring,
            Accumulation::Reduction(monoid) => {
                panic!("{site}: a block opened to reduce under {monoid:?} contracts nothing")
            }
        }
    }
}

/// An `mr × nr` block of `RA`-wide register accumulators, the software [`PlaneTile`].
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct RegisterData<T: Numeric> {
    /// `mr * nr` lines, each `Vector<T, RA>` (width registered in [`alloc`](Self::alloc)).
    pub(crate) data: Array<Vector<T, RA>>,
    /// Physical line width, the numeric twin of `RA`; comptime, for the line arithmetic.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Partials a line holds of one cell: `1` for neighbouring cells, else the line width.
    #[cube(comptime)]
    pub(crate) fold: usize,
    /// Rows in the block.
    #[cube(comptime)]
    pub(crate) mr: usize,
    /// Lines per row: `n / vector_size`, or `n` when folded.
    #[cube(comptime)]
    pub(crate) nr: usize,
    /// The sink's matrix this block was sized against.
    #[cube(comptime)]
    pub(crate) axes: MatrixAxes,
    /// Execution configuration for this register leaf.
    #[cube(comptime)]
    pub(crate) config: RegisterBlock,
    /// What this block accumulates, and so the `⊕` its partials merge under.
    #[cube(comptime)]
    pub(crate) accumulation: Accumulation,
}

/// Bind the block width `RA` for the rest of the kernel's scope.
#[cube]
fn register_block_size(#[comptime] vector_size: usize) {
    intrinsic!(|scope| {
        scope.register_size::<RA>(vector_size);
    });
}

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// An uninitialized `m × n` block of `vector_size`-wide lines, each `fold` partials of a cell.
    /// Unfolded, `n` must be a whole number of lines.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alloc(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] axes: MatrixAxes,
        #[comptime] vector_size: usize,
        #[comptime] fold: usize,
        #[comptime] config: RegisterBlock,
        #[comptime] accumulation: Accumulation,
    ) -> RegisterData<T> {
        comptime!(assert!(
            fold == 1 || fold == vector_size,
            "RegisterData::alloc: a line holds the {fold} partials of one cell, so it is that wide, \
             not {vector_size}"
        ));
        comptime!(assert!(
            vector_size > 0 && (fold > 1 || n.is_multiple_of(vector_size)),
            "RegisterData::alloc: n ({n}) must be a whole number of {vector_size}-wide lines"
        ));
        register_block_size(vector_size);
        let nr = comptime!(if fold > 1 { n } else { n / vector_size });
        RegisterData::<T> {
            data: Array::<Vector<T, RA>>::new(comptime!(m * nr)),
            vector_size,
            fold,
            mr: m,
            nr,
            axes,
            config,
            accumulation,
        }
    }

    pub(crate) fn zero(&mut self) {
        self.init(T::from_int(0));
    }

    pub(crate) fn init(&mut self, val: T) {
        let count = comptime!(self.mr * self.nr);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            self.data[i] = Vector::<T, RA>::cast_from(val);
        }
    }

    /// Add `src`'s block (same grid and fold) into this one, casting each line up to `T`.
    pub(crate) fn add_cast_from<S: Numeric>(&mut self, src: &RegisterData<S>) {
        comptime!(assert!(
            self.mr == src.mr && self.nr == src.nr && self.fold == src.fold,
            "RegisterData::add_cast_from: the blocks are {}x{} fold {} and {}x{} fold {}; a \
             promotion adds line for line and both are opened against one sink",
            self.mr,
            self.nr,
            self.fold,
            src.mr,
            src.nr,
            src.fold
        ));
        let count = comptime!(self.mr * self.nr);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            self.data[i] += Vector::<T, RA>::cast_from(src.data[i]);
        }
    }

    /// `self[i, :] *= factors[first + i]` for every slice `i` along the block's columns, in the
    /// registers that hold it.
    pub(crate) fn mul_along(&mut self, factors: &Array<T>, #[comptime] first: usize) {
        #[unroll]
        for r in 0..self.mr {
            let factor = Vector::<T, RA>::cast_from(factors[first + r]);
            #[unroll]
            for n in 0..self.nr {
                self.data[r * self.nr + n] *= factor;
            }
        }
    }

    /// Multiply every partial this block holds by `factor`.
    pub(crate) fn scale(&mut self, factor: T) {
        let count = comptime!(self.mr * self.nr);
        let line = Vector::<T, RA>::cast_from(factor);
        #[allow(clippy::needless_range_loop)]
        #[unroll]
        for i in 0..count {
            self.data[i] *= line;
        }
    }
}

#[cube]
impl<T: Numeric> RegisterData<T> {
    /// Write the block into `mem`'s window, casting down to its element.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        if comptime!(self.fold > 1) {
            let size!(A) = 1usize;
            self.drain::<Out, A>(mem, space);
        } else {
            self.drain::<Out, RA>(mem, space);
        }
    }

    /// [`store_cast_window`](Self::store_cast_window) into a sink of `A`-wide cells.
    fn drain<Out: Numeric, A: Size>(&self, mem: &mut Memory<Out>, #[comptime] space: Space) {
        let mem_write = comptime!(mem.access.write);
        let unit_share = comptime!(mem.unit_share);
        let fold = comptime!(self.fold);
        let monoid = comptime!(self.accumulation.monoid());
        let mut sink = mem.matrix_mut::<A>(0usize, comptime!(self.axes), space);

        // Split at comptime: a per-line `match` plus a unit guard breaks the CPU backend.
        match comptime!(Drain::of(unit_share, mem_write)) {
            Drain::EachUnit =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let cell =
                            cell::<T, Out, A>(self.data[comptime!(i * self.nr + n)], fold, monoid);
                        sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                    }
                }
            }
            Drain::UnitZero =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let cell =
                            cell::<T, Out, A>(self.data[comptime!(i * self.nr + n)], fold, monoid);
                        if UNIT_POS_X == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
            Drain::PlaneFold =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = UnitShare::Plane
                            .reduce::<Vector<T, RA>>(self.data[comptime!(i * self.nr + n)], monoid);
                        let cell = cell::<T, Out, A>(combined, fold, monoid);
                        if UNIT_POS_X == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
            Drain::GroupFold { unit_bits } =>
            {
                #[unroll]
                for i in 0..comptime!(self.mr) {
                    #[unroll]
                    for n in 0..comptime!(self.nr) {
                        let combined = comptime!(UnitShare::Group { unit_bits })
                            .reduce::<Vector<T, RA>>(self.data[comptime!(i * self.nr + n)], monoid);
                        let cell = cell::<T, Out, A>(combined, fold, monoid);
                        let unit_in_group = UNIT_POS_X & comptime!(unit_bits as u32);
                        if unit_in_group == 0 {
                            sink.write(((i as u32).runtime(), (n as u32).runtime()), cell);
                        }
                    }
                }
            }
        }
    }
}

/// A drained line cast to `Out`, first folded under `monoid` when it holds partials.
#[cube]
fn cell<T: Numeric, Out: Numeric, A: Size>(
    line: Vector<T, RA>,
    #[comptime] fold: usize,
    #[comptime] monoid: Monoid,
) -> Vector<Out, A> {
    if comptime!(fold > 1) {
        Vector::<Out, A>::cast_from(Monoid::reduce::<T, RA>(line, fold, monoid))
    } else {
        Vector::<Out, A>::cast_from(line)
    }
}

/// The block cut into slices along its columns, each a unit holds whole: the verbs
/// [`AxisSlices`] runs on a register block. The block holds whole cells, never a line of partials
/// of one.
#[cube]
impl<E: Float> RegisterData<E> {
    /// `self[i, s] = self[i, s] · scale + bias[i, s]` at every cell, the bias read at the cell of
    /// a tile over `space` that the block holds.
    pub(crate) fn scale_add_along(
        &mut self,
        scale: E,
        bias: &Procedural<E>,
        #[comptime] space: Space,
    ) {
        #[unroll]
        for r in 0..self.mr {
            #[unroll]
            for n in 0..self.nr {
                let at = r * self.nr + n;
                let mut line = self.data[at];
                #[unroll]
                for j in 0..self.vector_size {
                    let column = n * self.vector_size + j;
                    let cell = bias.value_at(r as u32, column as u32, space.clone());
                    line.insert(j, line.extract(j) * scale + cell);
                }
                self.data[at] = line;
            }
        }
    }

    /// Each slice's max, starting from `seed`'s.
    pub(crate) fn maxima_along(&self, seed: &Array<E>) -> Array<E> {
        let mut maxima = Array::<E>::new(self.mr);
        #[unroll]
        for r in 0..self.mr {
            let mut slice = seed[r];
            #[unroll]
            for n in 0..self.nr {
                let line = self.data[r * self.nr + n];
                #[unroll]
                for j in 0..self.vector_size {
                    slice = max(slice, line.extract(j));
                }
            }
            maxima[r] = slice;
        }
        maxima
    }

    /// `self[i, s] = exp(self[i, s] − slices[i])` ([`AxisSlices::exp_minus_cell`]).
    pub(crate) fn exp_minus_along(&mut self, slices: &Array<E>) {
        #[unroll]
        for r in 0..self.mr {
            #[unroll]
            for n in 0..self.nr {
                let at = r * self.nr + n;
                let mut line = self.data[at];
                #[unroll]
                for j in 0..self.vector_size {
                    line.insert(
                        j,
                        AxisSlices::<E>::exp_minus_cell(line.extract(j), slices[r]),
                    );
                }
                self.data[at] = line;
            }
        }
    }

    /// Each slice's sum.
    pub(crate) fn sums_along(&self) -> Array<E> {
        let mut sums = Array::<E>::new(self.mr);
        #[unroll]
        for r in 0..self.mr {
            let mut slice = E::from_int(0);
            #[unroll]
            for n in 0..self.nr {
                let line = self.data[r * self.nr + n];
                #[unroll]
                for j in 0..self.vector_size {
                    slice += line.extract(j);
                }
            }
            sums[r] = slice;
        }
        sums
    }
}
