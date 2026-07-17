//! The manual-mma encoding of a plane tile ([`MmaData`]) and its fragment↔memory transports.
//! The raw-mma twin of [`cmma`](super::cmma), issuing [`MmaDefinition::execute`] over per-lane
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
    pub io: MmaIOConfig,
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
        #[comptime] io: MmaIOConfig,
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
        #[comptime] io: MmaIOConfig,
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
        #[comptime] io: MmaIOConfig,
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

    /// Fill this fragment from `mem`'s window (row-major stage, scalar row stride), by the role's
    /// transport ([`Manual`](LoadMethod::Manual) index math or the `ldmatrix` intrinsic).
    pub(crate) fn load_window(&mut self, mem: &MemData<T>) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let layout = comptime!(self.layout);
        let io = comptime!(self.io);
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        match &mut self.fragment {
            MmaFragment::Lhs(f) => load_fragment(mem, f, &def, MatrixIdent::A, layout, io),
            MmaFragment::Rhs(f) => load_fragment(mem, f, &def, MatrixIdent::B, layout, io),
            MmaFragment::Acc(f) => {
                load_fragment(mem, f, &def, MatrixIdent::Accumulator, layout, io)
            }
        }
    }

    /// Drain this (accumulator) fragment into `mem`'s window.
    pub(crate) fn store_window(&self, mem: &mut MemData<T>) {
        self.store_cast_window::<T>(mem)
    }

    /// Drain this (accumulator) fragment into `mem`'s window, casting `T` down to the sink element.
    pub(crate) fn store_cast_window<Out: Numeric>(&self, mem: &mut MemData<Out>) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let layout = comptime!(self.layout);
        let io = comptime!(self.io);
        let def = MmaDefinition::<T, T, T>::new(m, n, k);
        match &self.fragment {
            MmaFragment::Acc(f) => store_fragment::<T, Out, T, T, T>(
                mem,
                f,
                &def,
                MatrixIdent::Accumulator,
                layout,
                io,
            ),
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
// Fill / load / store primitives (adapted from cubek-std's mma module to read/write cubek-tile's
// `MemData` window: a row-major stage addressed by `window_slice()` + scalar `row_stride()`).
// ===========================================================================

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

/// Load `fragment` (role `ident`) from `mem`'s row-major window. `ldmatrix` needs a vectorized
/// row slice, which a `MemData` window cannot serve yet, so that path is refused rather than wrong.
#[cube]
fn load_fragment<T: Numeric, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    mem: &MemData<T>,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIOConfig,
) {
    match io.load_method(ident) {
        LoadMethod::Manual => load_manual(mem, fragment, def, ident, layout),
        LoadMethod::LoadMatrix => {
            comptime!(panic!(
                "MmaData::load: the ldmatrix fast path is not yet wired for MemData windows; \
                 build the Leaf with MmaIOConfig::manual()"
            ))
        }
    }
}

/// Manual load: for each register the hardware position `(row, col)` of its element(s), read from
/// the row-major window (`offset = row · row_stride + col`).
#[cube]
fn load_manual<T: Numeric, N: Size, A: Numeric, B: Numeric, CD: Numeric>(
    mem: &MemData<T>,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let window = mem.window_slice();
    let stride = mem.row_stride();
    let (stride_row, stride_col) = match comptime!(layout) {
        MatrixLayout::RowMajor => (stride, 1u32),
        MatrixLayout::ColMajor => (1u32, stride),
        MatrixLayout::Undefined => panic!("mma: a stage layout must be row- or col-major"),
    };

    #[unroll]
    for i in 0..num_vectors {
        let mut vector = Vector::empty();
        #[unroll]
        for e in 0..vector_size {
            let elem_idx = i * vector_size + e;
            let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
            let offset = row * stride_row + col * stride_col;
            vector.insert(e, T::cast_from(window[offset as usize]));
        }
        fragment[i] = vector;
    }
}

/// Store `fragment` (accumulator) into `mem`'s window, casting to `Out`. `stmatrix` is refused
/// like `ldmatrix` (see [`load_fragment`]).
#[cube]
fn store_fragment<T: Numeric, Out: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    mem: &mut MemData<Out>,
    fragment: &Array<Vector<T, NA>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIOConfig,
) {
    match io.store_method() {
        StoreMethod::Manual => store_manual::<T, Out, A, B, CD>(mem, fragment, def, ident, layout),
        StoreMethod::StoreMatrix => {
            comptime!(panic!(
                "MmaData::store: the stmatrix fast path is not yet wired for MemData windows; \
                 build the Leaf with MmaIOConfig::manual()"
            ))
        }
    }
}

/// Manual store: each register's element(s) written to their hardware position in the row-major
/// window, cast to the sink element.
#[cube]
fn store_manual<T: Numeric, Out: Numeric, A: Numeric, B: Numeric, CD: Numeric>(
    mem: &mut MemData<Out>,
    fragment: &Array<Vector<T, NA>>,
    def: &MmaDefinition<A, B, CD>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
) {
    let num_vectors = def.vectors_per_lane(ident);
    let vector_size = def.vector_size(ident);
    let lane_id = UNIT_POS_PLANE;

    let stride = mem.row_stride();
    let (stride_row, stride_col) = match comptime!(layout) {
        MatrixLayout::RowMajor => (stride, 1u32),
        MatrixLayout::ColMajor => (1u32, stride),
        MatrixLayout::Undefined => panic!("mma: a stage layout must be row- or col-major"),
    };
    let window = mem.window_slice_mut();

    #[unroll]
    for i in 0..num_vectors {
        #[unroll]
        for e in 0..vector_size {
            let elem_idx = i * vector_size + e;
            let (row, col) = def.position_of_nth(lane_id, elem_idx as u32, ident);
            let offset = row * stride_row + col * stride_col;
            window[offset as usize] = Out::cast_from(fragment[i].extract(e));
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
