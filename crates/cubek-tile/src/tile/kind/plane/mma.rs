//! The manual-mma encoding of a plane tile ([`MmaData`]) and its fragment↔memory transports.
//! The raw-mma twin of [`cmma`](super::cmma), issuing [`MmaDefinition::execute`] over per-unit
//! registers instead of `cmma::execute`.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    prelude::*,
};

use crate::*;

// Per-role fragment register widths, bound at allocation via `scope.register_size` to match
// `def.vector_size(role)`. Independent of the outer tile's stage vector width.
define_size!(pub NL);
define_size!(pub NR);
define_size!(pub NA);

/// A single manual-mma fragment: the register array for one role plus the comptime shape/transport
/// its load/store paths dispatch on. `Clone` duplicates the handle, not the registers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MmaData<T: Numeric> {
    pub fragment: MmaFragment<T>,
    #[cube(comptime)]
    pub m: usize,
    #[cube(comptime)]
    pub n: usize,
    #[cube(comptime)]
    pub k: usize,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    #[cube(comptime)]
    pub io: MmaIo,
}

/// One role's register array. The role rides inside the fragment because each carries a different
/// inner width (`NL`/`NR`/`NA`).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum MmaFragment<T: Numeric> {
    Lhs(Array<Vector<T, NL>>),
    Rhs(Array<Vector<T, NR>>),
    Acc(Array<Vector<T, NA>>),
}

#[cube]
impl<T: Numeric> MmaData<T> {
    /// Allocate an accumulator fragment. A role's register width depends only on its own element
    /// type, so `<T,T,T>` gives correct `Accumulator` metadata without the real `(L,R,A)` triple.
    pub(crate) fn acc(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
    ) -> MmaData<T> {
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        register_acc_size::<T>(&def);
        let count = def.vectors_per_lane(MatrixIdent::Accumulator);
        MmaData::<T> {
            fragment: MmaFragment::new_Acc(Array::new(count)),
            m,
            n,
            k,
            layout,
            io,
        }
    }

    /// Allocate an `A`-role (lhs) operand fragment.
    pub(crate) fn lhs(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
    ) -> MmaData<T> {
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        register_lhs_size::<T>(&def);
        let count = def.vectors_per_lane(MatrixIdent::A);
        MmaData::<T> {
            fragment: MmaFragment::new_Lhs(Array::new(count)),
            m,
            n,
            k,
            layout,
            io,
        }
    }

    /// Allocate a `B`-role (rhs) operand fragment.
    pub(crate) fn rhs(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
    ) -> MmaData<T> {
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        register_rhs_size::<T>(&def);
        let count = def.vectors_per_lane(MatrixIdent::B);
        MmaData::<T> {
            fragment: MmaFragment::new_Rhs(Array::new(count)),
            m,
            n,
            k,
            layout,
            io,
        }
    }

    /// Zero the fragment, whatever the role.
    pub(crate) fn zero(&mut self) {
        match &mut self.fragment {
            MmaFragment::Lhs(f) => fill_registers(f, T::from_int(0)),
            MmaFragment::Rhs(f) => fill_registers(f, T::from_int(0)),
            MmaFragment::Acc(f) => fill_registers(f, T::from_int(0)),
        }
    }

    /// Fill this fragment from `src`'s window (row-major stage), by the role's transport
    /// ([`Manual`](LoadMethod::Manual) index math or the `ldmatrix` intrinsic). Takes the tile, not
    /// its store: the manual path reads through the quant-transparent matrix view, decoding here.
    pub(crate) fn load_window(&mut self, src: &Tile<T>) {
        let dequant_at = src.dequant_at();
        let io = comptime!(self.io);
        comptime!(assert!(
            dequant_at == DequantAt::Load
                || (matches!(io.lhs_load_method, LoadMethod::Manual)
                    && matches!(io.rhs_load_method, LoadMethod::Manual)),
            "MmaData::load_window: the ldmatrix transport copies raw units, so it cannot decode a \
             quantized source as it reads; serve that operand by its load (DequantAt::Load)"
        ));
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let layout = comptime!(self.layout);
        let io = comptime!(self.io);
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        match &mut self.fragment {
            MmaFragment::Lhs(f) => load_fragment(src, f, &def, MatrixIdent::A, layout, io, (m, k)),
            MmaFragment::Rhs(f) => load_fragment(src, f, &def, MatrixIdent::B, layout, io, (k, n)),
            MmaFragment::Acc(f) => {
                load_fragment(src, f, &def, MatrixIdent::Accumulator, layout, io, (m, n))
            }
        }
    }

    /// Drain this (accumulator) fragment into `mem`'s window, `space` being the window's.
    pub(crate) fn store_window(&self, mem: &mut Memory<T>, #[comptime] space: Space) {
        self.store_cast_window::<T>(mem, space)
    }

    /// Drain this (accumulator) fragment into `mem`'s window, `space` being the window's, casting
    /// `T` down to the sink element: each unit writing its own cells through the destination's
    /// write, which masks a cell past the window's edge and adds into a destination that folds. A
    /// unit knows which cells it holds, so every cell has one writer and no scratch is needed to
    /// elect it. The `stmatrix` transport does not reach a memory window, so the store is always
    /// this one, whatever [`MmaIo::store_method`] says.
    pub(crate) fn store_cast_window<Out: Numeric>(
        &self,
        mem: &mut Memory<Out>,
        #[comptime] space: Space,
    ) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let layout = comptime!(self.layout);
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        match &self.fragment {
            MmaFragment::Acc(f) => store_cells::<T, Out, T, T, T>(mem, f, &def, layout, space),
            MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                panic!("MmaData::store: only an accumulator fragment drains to memory")
            }
        }
    }
}

// ===========================================================================
// Register-size binding
// ===========================================================================

#[cube]
fn register_acc_size<A: Numeric>(def: &MmaDefinition<A, A, A>) {
    let va = def.vector_size(MatrixIdent::Accumulator);
    intrinsic!(|scope| {
        scope.register_size::<NA>(va);
    });
}

#[cube]
fn register_lhs_size<L: Numeric>(def: &MmaDefinition<L, L, L>) {
    let vl = def.vector_size(MatrixIdent::A);
    intrinsic!(|scope| {
        scope.register_size::<NL>(vl);
    });
}

#[cube]
fn register_rhs_size<R: Numeric>(def: &MmaDefinition<R, R, R>) {
    let vr = def.vector_size(MatrixIdent::B);
    intrinsic!(|scope| {
        scope.register_size::<NR>(vr);
    });
}

// ===========================================================================
// Fill / load / store primitives over cubek-tile's `Memory` window: a row-major stage addressed by
// `window_slice()` + scalar `row_stride()`. (Adapted from cubek-std's mma module.)

/// Fill every register slot with `value`.
#[cube]
fn fill_registers<E: Numeric, N: Size>(fragment: &mut Array<Vector<E, N>>, value: E) {
    let num_vectors = fragment.len();
    let v = Vector::<E, N>::cast_from(value);
    #[unroll]
    for i in 0..num_vectors {
        fragment[i] = v;
    }
}

/// Load `fragment` (role `ident`, shaped by `edges`) from a row-major window.
#[cube]
fn load_fragment<T: Numeric, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    src: &Tile<T>,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] edges: (usize, usize),
) {
    // Fall back to manual loading for gathered operands.
    let gathered = src.gathered();
    let method = comptime!(if gathered {
        LoadMethod::Manual
    } else {
        io.load_method(ident)
    });
    match method {
        LoadMethod::Manual => {
            let size!(W) = src.vector_size();
            load_manual::<T, W, N, A, B, CD>(src, fragment, def, ident, layout, edges)
        }
        LoadMethod::LoadMatrix => {
            comptime!(panic!(
                "MmaData::load: the ldmatrix fast path is not yet wired for Memory windows; \
                 state the register stage with MmaIo::manual()"
            ))
        }
    }
}

/// Manual load: each lane reads its own cells out of `src` through the matrix view, the window
/// lying as the role's `edges` (`layout` row-major) or as their transpose (col-major, a weight
/// stored `{n, k}`).
///
/// Where a lane's register vector runs along the window's lines — the instruction's
/// [`vector_layout`](MmaDefinition::vector_layout) is the window's — the lane reads whole lines,
/// one per `W` cells, rather than one line per cell: a load the device issues wide. Elsewhere
/// each cell is read on its own.
#[cube]
fn load_manual<T: Numeric, W: Size, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    src: &Tile<T>,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] edges: (usize, usize),
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let vector_layout = def.vector_layout(ident);
    let unit_id = UNIT_POS_PLANE;
    let served = src.vector_size();
    let width = comptime!(served);
    let (rows, cols) = comptime!(edges);
    let transposed = comptime!(match layout {
        MatrixLayout::RowMajor => false,
        MatrixLayout::ColMajor => true,
        MatrixLayout::Undefined => {
            panic!("MmaData::load: a manual fragment load reads a row- or col-major window")
        }
    });
    // A line is addressed by its index within the window, so the window's edge along its lines
    // must hold whole lines: a fragment narrower than a line (NVIDIA's `n = 8` beside a 16-wide
    // line) starts inside one, and its index would round down to the line before.
    let line_edge = comptime!(if transposed { rows } else { cols });
    comptime!(assert!(
        line_edge.is_multiple_of(width),
        "MmaData::load: a {width}-wide line runs past the fragment's {line_edge}-cell edge along \
         it, which a manual fragment load cannot read from inside the line; serve the operand at \
         most {line_edge} wide"
    ));
    let view = if comptime!(transposed) {
        src.fragment_matrix_packed::<W>(cols, rows)
    } else {
        src.fragment_matrix_packed::<W>(rows, cols)
    };
    let along_lines = comptime!(vector_layout == layout);
    comptime!(assert!(
        !along_lines || vector_size.is_multiple_of(width) || width.is_multiple_of(vector_size),
        "MmaData::load: a {vector_size}-cell register vector neither holds whole {width}-wide \
         lines nor sits inside one"
    ));

    #[unroll]
    for i in 0..num_vectors {
        let mut vector = Vector::empty();
        if comptime!(along_lines) {
            // The vector's cells are consecutive along the window's line axis, from its first.
            let (row, col) = def.position_of_nth(unit_id, comptime!(i * vector_size) as u32, ident);
            let (line_row, cell) = if comptime!(transposed) {
                (col, row)
            } else {
                (row, col)
            };
            if comptime!(width >= vector_size) {
                // One line holds the whole vector: a register vector of a matrix instruction
                // starts at a multiple of its own size along the axis it runs down (the unit's
                // share of a row or column is whole vectors), and `width` is a multiple of
                // `vector_size`, so `start + vector_size` never passes `width`.
                let line = view.read((line_row, cell / comptime!(width as u32)));
                let start = cell % comptime!(width as u32);
                #[unroll]
                for e in 0..vector_size {
                    vector.insert(e, line.extract_dynamic((start + e as u32).cast::<usize>()));
                }
            } else {
                #[unroll]
                for l in 0..comptime!(vector_size / width) {
                    let line = view.read((
                        line_row,
                        cell / comptime!(width as u32) + comptime!(l as u32),
                    ));
                    #[unroll]
                    for e in 0..width {
                        vector.insert(comptime!(l * width + e), line.extract(e));
                    }
                }
            }
        } else {
            #[unroll]
            for e in 0..vector_size {
                let elem_idx = i * vector_size + e;
                let (row, col) = def.position_of_nth(unit_id, elem_idx as u32, ident);
                let (line_row, cell) = if comptime!(transposed) {
                    (col, row)
                } else {
                    (row, col)
                };
                let line = view.read((line_row, cell / comptime!(width as u32)));
                vector.insert(
                    e,
                    line.extract_dynamic((cell % comptime!(width as u32)).cast::<usize>()),
                );
            }
        }
        fragment[i] = vector;
    }
}

/// Each unit's accumulator cells written through `mem`'s own write, one element at a time: the
/// write masks a cell past the window's edge and adds into a destination that folds. A cell is one
/// unit's, so each is written once.
#[cube]
fn store_cells<T: Numeric, Out: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    mem: &mut Memory<Out>,
    fragment: &Array<Vector<T, NA>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] layout: MatrixLayout,
    #[comptime] space: Space,
) {
    comptime!(assert!(
        mem.store.vector_size == 1,
        "MmaData: a fragment drained cell by cell writes one element at a time, and its \
         destination is bound {} wide; bind it one element wide",
        mem.store.vector_size
    ));
    let num_vectors = def.vectors_per_lane(MatrixIdent::Accumulator);
    let vector_size = def.vector_size(MatrixIdent::Accumulator);
    let unit_id = UNIT_POS_PLANE;
    let axes = comptime!(MatrixAxes::trailing(&space));
    let mut sink = mem.matrix_mut::<Const<1>>(0usize, axes, space);

    #[unroll]
    for i in 0..num_vectors {
        #[unroll]
        for e in 0..vector_size {
            let elem_idx = i * vector_size + e;
            let (row, col) =
                def.position_of_nth(unit_id, elem_idx as u32, MatrixIdent::Accumulator);
            let at = match comptime!(layout) {
                MatrixLayout::RowMajor => (row, col),
                MatrixLayout::ColMajor => (col, row),
                MatrixLayout::Undefined => panic!("mma: a stage layout must be row- or col-major"),
            };
            let mut value = Vector::<Out, Const<1>>::empty();
            value.insert(0usize, Out::cast_from(fragment[i].extract(e)));
            sink.write(at, value);
        }
    }
}

/// Execute `acc += lhs · rhs` over three role fragments via the manual `MmaDefinition::execute`,
/// copying the result registers back into `acc`.
#[cube]
pub(crate) fn mma_execute<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NL>>,
    rhs: &Array<Vector<R, NR>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let def = MmaDefinition::<L, R, A>::new(m, n, k);
    let out = def.execute(lhs, rhs, &*acc);
    let num = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num {
        acc[i] = out[i];
    }
}
