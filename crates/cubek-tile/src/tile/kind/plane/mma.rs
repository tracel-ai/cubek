//! The manual-mma encoding of a plane tile ([`MmaData`]) and its fragment↔memory transports.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2, e4m3,
    features::ScaledMmaConfig,
    ir::{ElemType, FloatKind},
    prelude::*,
    quant::scheme::QuantValue,
    std::tensor::layout::CoordsDyn,
};

use super::load_matrix::{LDMATRIX_ROW_BYTES, load_ldmatrix};
use crate::*;

// Per-role fragment register widths, bound at allocation to `def.vector_size(role)`.
define_size!(pub(crate) NL);
define_size!(pub(crate) NR);
define_size!(pub(crate) NA);
// A block-scaled fragment's widths: its values' per role, and its scales'.
define_size!(pub(crate) NLB);
define_size!(pub(crate) NRB);
define_size!(pub(crate) NSB);

/// `e2m1` values one stored word holds, and so one register of the block-scaled instruction.
const E2M1_PER_WORD: usize = 8;

/// Words one `ldmatrix` row holds.
const LDMATRIX_ROW_WORDS: usize = LDMATRIX_ROW_BYTES / size_of::<u32>();

/// Values one `e2m1` block scale covers along the contraction under the instruction this
/// encoding runs: NVFP4's block, a `ue4m3` scale every sixteen values.
pub(crate) const E2M1_SCALE_BLOCK: usize = 16;

/// One manual-mma fragment: a role's registers plus the shape and transport it dispatches on.
/// `Clone` duplicates the handle, not the registers.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct MmaData<T: Numeric> {
    pub(crate) fragment: MmaFragment<T>,
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

/// One role's register array, each role at its own width (`NL`/`NR`/`NA`).
#[expect(
    dead_code,
    reason = "built through the expand type's generated constructors"
)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) enum MmaFragment<T: Numeric> {
    Lhs(Array<Vector<T, NL>>),
    Rhs(Array<Vector<T, NR>>),
    Acc(Array<Vector<T, NA>>),
    /// A block-scaled `A`: its `e2m1` values as the instruction takes them, two a byte, and the
    /// `e4m3` scales of the row this unit's scale register serves, the one register an array of
    /// one holds, so a handle to the fragment loads the registers the fragment holds.
    LhsBlockScaled(Array<Vector<e2m1x2, NLB>>, Array<Vector<e4m3, NSB>>),
    /// A block-scaled `B`, as [`LhsBlockScaled`](Self::LhsBlockScaled) is an `A`.
    RhsBlockScaled(Array<Vector<e2m1x2, NRB>>, Array<Vector<e4m3, NSB>>),
}

#[cube]
impl<T: Numeric> MmaData<T> {
    /// Allocate an accumulator fragment.
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

    /// Allocate an operand fragment in role `ident`: block-scaled where `block_scaled` says the
    /// operand contracts through the device's block-scaled instruction ([`block_scales_here`]).
    pub(crate) fn operand(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
        #[comptime] block_scaled: bool,
    ) -> MmaData<T> {
        match comptime!((ident, block_scaled)) {
            (_, true) => MmaData::<T>::block_scaled(ident, m, n, k, layout, io),
            (MatrixIdent::A, false) => MmaData::<T>::lhs(m, n, k, layout, io),
            (MatrixIdent::B, false) => MmaData::<T>::rhs(m, n, k, layout, io),
            (MatrixIdent::Accumulator, false) => {
                panic!("MmaData::operand: an accumulator is not an operand")
            }
        }
    }

    /// Allocate a block-scaled operand fragment in role `ident`: `e2m1` values under an `e4m3`
    /// scale every [`E2M1_SCALE_BLOCK`] of them along `k`, the device's block-scaled instruction
    /// of `m × n × k` ([`block_scales_here`]).
    pub(crate) fn block_scaled(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] io: MmaIo,
    ) -> MmaData<T> {
        let def = block_scaled_definition::<f32>(m, n, k);
        register_block_scaled_sizes(&def, ident);
        let fragment = match comptime!(ident) {
            MatrixIdent::A => MmaFragment::new_LhsBlockScaled(
                Array::new(def.vectors_per_lane(MatrixIdent::A)),
                Array::new(1usize),
            ),
            MatrixIdent::B => MmaFragment::new_RhsBlockScaled(
                Array::new(def.vectors_per_lane(MatrixIdent::B)),
                Array::new(1usize),
            ),
            MatrixIdent::Accumulator => {
                panic!("MmaData::block_scaled: an accumulator carries no scales")
            }
        };
        MmaData::<T> {
            fragment,
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
            MmaFragment::LhsBlockScaled(..) | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::zero: a block-scaled operand is loaded, never zeroed")
            }
        }
    }

    /// Multiply every cell of this (accumulator) fragment by `factor`: its registers are the
    /// unit's own cells, so the scale needs no bounce.
    pub(crate) fn scale(&mut self, factor: T) {
        match &mut self.fragment {
            MmaFragment::Acc(f) => scale_registers(f, factor),
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::scale: an operand fragment is contracted, not scaled")
            }
        }
    }

    /// Fill this fragment from `src`'s window by the role's transport.
    pub(crate) fn load_window(&mut self, src: &Tile<T>) {
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
            MmaFragment::LhsBlockScaled(values, scales) => {
                load_block_scaled(src, values, scales, MatrixIdent::A, layout, io, (m, n, k))
            }
            MmaFragment::RhsBlockScaled(values, scales) => {
                load_block_scaled(src, values, scales, MatrixIdent::B, layout, io, (m, n, k))
            }
        }
    }

    /// Drain this (accumulator) fragment into `mem`'s window, `space` being the window's.
    pub(crate) fn store_window(&self, mem: &mut Memory<T>, #[comptime] space: Space) {
        self.store_cast_window::<T>(mem, space)
    }

    /// Drain this (accumulator) fragment into `mem`'s window, casting down to the sink element.
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
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::store: only an accumulator fragment drains to memory")
            }
        }
    }
}

/// An accumulator read where its units hold it: each register vector runs along one row of the
/// fragment, and a row's cells lie across the units [`MmaDefinition::position_of_nth`] says share
/// it. Slot `first + i` of the arrays these take or return is the row of the unit's register
/// vector `i`, so each unit keeps the state of its own rows and no others.
#[cube]
impl<E: Numeric> MmaData<E> {
    /// Rows of this accumulator one unit holds cells of, one per register vector.
    pub(crate) fn rows_per_unit(&self) -> comptime_type!(usize) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        def.vectors_per_lane(MatrixIdent::Accumulator)
    }

    /// `self[r, c] = self[r, c] · scale + bias[r, c]`, `bias` read at the cell's coordinates in
    /// the window over `space`, this fragment's first cell at `origin` of it.
    pub(crate) fn scale_add_along(
        &mut self,
        scale: E,
        bias: &Procedural<E>,
        #[comptime] space: Space,
        #[comptime] origin: (usize, usize),
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let (row0, col0) = comptime!(origin);
        let acc = self.acc_registers_mut();
        #[unroll]
        for i in 0..acc.len() {
            let mut vector = acc[i];
            #[unroll]
            for e in 0..vector_size {
                let (row, col) = def.position_of_nth(
                    UNIT_POS_PLANE,
                    comptime!((i * vector_size + e) as u32),
                    MatrixIdent::Accumulator,
                );
                let cell = bias.value_at(
                    row + comptime!(row0 as u32),
                    col + comptime!(col0 as u32),
                    comptime!(space.clone()),
                );
                vector.insert(e, vector.extract(e) * scale + cell);
            }
            acc[i] = vector;
        }
    }

    /// `self[r, :] *= factors[first + r]` for every row `r` this unit holds cells of.
    pub(crate) fn mul_along(&mut self, factors: &Array<E>, #[comptime] first: usize) {
        let acc = self.acc_registers_mut();
        #[unroll]
        for i in 0..acc.len() {
            acc[i] *= Vector::cast_from(factors[first + i]);
        }
    }

    /// `slots[first + r]` combined under `monoid` with this unit's cells on its row `r`: the
    /// unit's own part of each row, which [`fold_across_units`](Self::fold_across_units)
    /// completes.
    pub(crate) fn fold_along(
        &self,
        slots: &mut Array<E>,
        #[comptime] first: usize,
        #[comptime] monoid: Monoid,
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let acc = self.acc_registers();
        #[unroll]
        for i in 0..acc.len() {
            let vector = acc[i];
            let mut slot = slots[first + i];
            #[unroll]
            for e in 0..vector_size {
                slot = monoid.combine::<E>(slot, vector.extract(e));
            }
            slots[first + i] = slot;
        }
    }

    /// Each row's slot, from `first` on, combined under `monoid` over the units that share the
    /// row: a butterfly over the lane bits that leave the row in place.
    ///
    /// Whether a lane bit moves a row is read at lane 0, where both positions are constants the
    /// compiler folds, so a bit that moves it costs no shuffle. That holds for a layout whose row
    /// is a function of some of the lane's bits, as the manual-mma layouts are; the register
    /// attention's component test checks it on every lane.
    pub(crate) fn fold_across_units(
        &self,
        slots: &mut Array<E>,
        #[comptime] first: usize,
        #[comptime] monoid: Monoid,
    ) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let rows = def.vectors_per_lane(MatrixIdent::Accumulator);
        let width = mma_plane_width();
        let lane_bits = comptime!(width.trailing_zeros() as usize);
        #[unroll]
        for i in 0..rows {
            let nth = comptime!((i * vector_size) as u32);
            let (row, _) = def.position_of_nth(0u32, nth, MatrixIdent::Accumulator);
            let mut value = slots[first + i];
            #[unroll]
            for bit in 0..lane_bits {
                let mask = comptime!(1u32 << bit);
                let (other_row, _) = def.position_of_nth(mask, nth, MatrixIdent::Accumulator);
                if other_row == row {
                    value = monoid.combine::<E>(value, plane_shuffle_xor(value, mask));
                }
            }
            slots[first + i] = value;
        }
    }

    fn acc_registers(&self) -> &Array<Vector<E, NA>> {
        match &self.fragment {
            MmaFragment::Acc(f) => f,
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData: an operand fragment's cells are contracted, not read by rows")
            }
        }
    }

    fn acc_registers_mut(&mut self) -> &mut Array<Vector<E, NA>> {
        match &mut self.fragment {
            MmaFragment::Acc(f) => f,
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData: an operand fragment's cells are contracted, not read by rows")
            }
        }
    }
}

#[cube]
impl<A: Numeric> MmaData<A> {
    /// This accumulator with each cell cast to `E` in the register that holds it.
    pub(crate) fn cast<E: Numeric>(&self) -> MmaData<E> {
        let acc = self.acc_registers();
        let mut cast = MmaData::<E>::acc(
            comptime!(self.m),
            comptime!(self.n),
            comptime!(self.k),
            comptime!(self.layout),
            comptime!(self.io),
        );
        let registers = cast.acc_registers_mut();
        #[unroll]
        for i in 0..acc.len() {
            registers[i] = Vector::cast_from(acc[i]);
        }
        cast
    }
}

#[cube]
impl<E: Float> MmaData<E> {
    /// Each row's max, starting from `seed`'s, over the units that share it.
    pub(crate) fn maxima_along(&self, seed: &Array<E>) -> Array<E> {
        let rows = self.rows_per_unit();
        let mut maxima = Array::<E>::new(rows);
        #[unroll]
        for i in 0..rows {
            maxima[i] = seed[i];
        }
        self.fold_along(&mut maxima, 0usize, Monoid::Max);
        self.fold_across_units(&mut maxima, 0usize, Monoid::Max);
        maxima
    }

    /// Each row's sum over the units that share it.
    pub(crate) fn sums_along(&self) -> Array<E> {
        let rows = self.rows_per_unit();
        let mut sums = Array::<E>::new(rows);
        #[unroll]
        for i in 0..rows {
            sums[i] = E::from_int(0);
        }
        self.fold_along(&mut sums, 0usize, Monoid::Sum);
        self.fold_across_units(&mut sums, 0usize, Monoid::Sum);
        sums
    }

    /// `self[r, c] = exp(self[r, c] − slices[first + r])` ([`AxisSlices::exp_minus_cell`]).
    pub(crate) fn exp_minus_along(&mut self, slices: &Array<E>, #[comptime] first: usize) {
        let def = MmaDefinition::<E, E, E>::new(self.m, self.n, self.k);
        let vector_size = def.vector_size(MatrixIdent::Accumulator);
        let acc = self.acc_registers_mut();
        #[unroll]
        for i in 0..acc.len() {
            let mut vector = acc[i];
            #[unroll]
            for e in 0..vector_size {
                vector.insert(
                    e,
                    AxisSlices::<E>::exp_minus_cell(vector.extract(e), slices[first + i]),
                );
            }
            acc[i] = vector;
        }
    }
}

/// The plane width the manual-mma layouts are stated for, read off the target at expansion.
#[cube]
fn mma_plane_width() -> comptime_type!(u32) {
    intrinsic!(|scope| scope.state().target_properties.mma.const_plane_size)
}

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

/// Bind a block-scaled fragment's widths in role `ident`: its values' and its scales'.
#[cube]
fn register_block_scaled_sizes(
    def: &MmaDefinition<e2m1x2, e2m1x2, f32>,
    #[comptime] ident: MatrixIdent,
) {
    let values = def.vector_size(ident);
    let scales = def.scales_vector_size();
    intrinsic!(|scope| {
        match ident {
            MatrixIdent::A => scope.register_size::<NLB>(values),
            MatrixIdent::B => scope.register_size::<NRB>(values),
            MatrixIdent::Accumulator => {
                panic!("MmaData::block_scaled: an accumulator carries no scales")
            }
        }
        scope.register_size::<NSB>(scales);
    });
}

/// Multiply every register slot by `factor`.
#[cube]
fn scale_registers<E: Numeric, N: Size>(fragment: &mut Array<Vector<E, N>>, factor: E) {
    let num_vectors = fragment.len();
    let factor = Vector::<E, N>::cast_from(factor);
    #[unroll]
    for i in 0..num_vectors {
        fragment[i] *= factor;
    }
}

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
    // `ldmatrix` reads 16-byte rows of 16-bit served cells from shared memory, on a device that
    // offers it for the element; any other window, or device, takes the manual load.
    let offered = loads_matrix::<T>();
    let gathered = src.gathered();
    let shared = src.is_shared();
    let element = src.stage_element();
    let holds_served_values = comptime!(element == StageElement::Served);
    let served = src.vector_size();
    // An element's size read at expansion, where the launch has registered it: inside
    // `comptime!` the call would size the generic placeholder instead.
    let elem_size = T::size().comptime();
    let row_cells = comptime!(LDMATRIX_ROW_BYTES / elem_size);
    let ldmatrix_serves = comptime!(
        offered
            && shared
            && !gathered
            && holds_served_values
            && elem_size == 2
            && ident != MatrixIdent::Accumulator
            && row_cells.is_multiple_of(served)
    );
    let method = comptime!(match ldmatrix_serves {
        true => io.load_method(ident),
        false => LoadMethod::Manual,
    });
    match method {
        LoadMethod::Manual => {
            let size!(W) = src.vector_size();
            load_manual::<T, W, N, A, B, CD>(src, fragment, def, ident, layout, edges)
        }
        LoadMethod::LoadMatrix => {
            let size!(W) = src.vector_size();
            load_ldmatrix::<T, W, N, A, B, CD>(src, fragment, def, ident, layout, edges)
        }
    }
}

/// Whether the device offers `ldmatrix` for `T`, read off its properties at expansion.
// `T` is read by the expansion, which is where a type has its element.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
fn loads_matrix<T: Numeric>() -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let element = T::elem_type(scope);
        scope
            .state()
            .device_properties
            .as_ref()
            .is_some_and(|properties| properties.features.matmul.ldmatrix.contains(&element))
    })
}

/// Manual load: each unit reads its own cells of `src` through the matrix view.
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
    // The window's edge along its lines must hold whole lines, or a line index rounds down.
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
            let (row, col) = def.position_of_nth(unit_id, comptime!(i * vector_size) as u32, ident);
            let (line_row, cell) = if comptime!(transposed) {
                (col, row)
            } else {
                (row, col)
            };
            if comptime!(width >= vector_size) {
                // A register vector starts at a multiple of its size, so it stays within `width`.
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

/// Each unit's accumulator cells written one element at a time through `mem`'s own write.
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

/// `acc += lhs · rhs` over three role fragments via `MmaDefinition::execute`.
/// An `m × k` accumulator's registers regrouped as the `A` operand of an `m × n × k` contraction:
/// for 16-bit operands the manual-mma layouts put the accumulator's cell `e` where the operand's
/// cell `e` is, so a product contracting its own output reads it without moving a cell between
/// units. A wider operand's `A` lays its cells otherwise (a `tf32` one along the columns first),
/// and is refused.
#[cube]
pub(crate) fn accumulator_as_a<L: Numeric>(
    acc: &Array<Vector<L, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> Array<Vector<L, NL>> {
    let acc_def = MmaDefinition::<L, L, L>::new(m, k, k);
    let a_def = MmaDefinition::<L, L, L>::new(m, n, k);
    register_lhs_size::<L>(&a_def);
    let acc_regs = acc_def.vectors_per_lane(MatrixIdent::Accumulator);
    let acc_width = acc_def.vector_size(MatrixIdent::Accumulator);
    let a_regs = a_def.vectors_per_lane(MatrixIdent::A);
    let a_width = a_def.vector_size(MatrixIdent::A);
    let bytes = L::size().comptime();
    comptime!(assert!(
        bytes == 2,
        "accumulator_as_a: an accumulator's cells sit where the A operand's do for 16-bit \
         operands; a {bytes}-byte operand's A lays them otherwise"
    ));
    comptime!(assert!(
        acc_regs * acc_width == a_regs * a_width,
        "accumulator_as_a: an {m}x{k} accumulator holds {} cells a unit, and the A operand of an \
         {m}x{n}x{k} contraction {}",
        acc_regs * acc_width,
        a_regs * a_width
    ));
    let mut registers = Array::<Vector<L, NL>>::new(a_regs);
    #[unroll]
    for i in 0..acc_regs {
        #[unroll]
        for e in 0..acc_width {
            let (register, slot) =
                comptime!(((i * acc_width + e) / a_width, (i * acc_width + e) % a_width));
            let mut vector = registers[register];
            vector.insert(slot, acc[i].extract(e));
            registers[register] = vector;
        }
    }
    registers
}

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

/// The device's block-scaled instruction of `m × n × k` over `e2m1` operands under `e4m3` scales,
/// summing in `CD`: one scale every [`E2M1_SCALE_BLOCK`] values of `k`.
#[cube]
fn block_scaled_definition<CD: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> MmaDefinition<e2m1x2, e2m1x2, CD> {
    MmaDefinition::<e2m1x2, e2m1x2, CD>::new_scaled::<e4m3>(
        m,
        n,
        k,
        comptime!(k / E2M1_SCALE_BLOCK),
    )
}

/// Whether `src`, packed `e2m1` values under one level of scales, contracts through the device's
/// block-scaled instruction of `m × n × k`: the instruction offered for `e2m1` operands under
/// `e4m3` scales, and the scales one every [`E2M1_SCALE_BLOCK`] values of `k`. A source that is
/// not reads through the decoding landing, which serves every scaled source.
///
/// What the instruction takes is NVFP4: the leaf reads each block's scale at its first value and
/// hands it over as `e4m3`, so the scales must cover sixteen values of `k` or a multiple of
/// sixteen, each exact in `e4m3`. A per-tensor factor above them is the product's to apply: the
/// instruction holds one level.
#[cube]
pub(crate) fn block_scales_here<T: Numeric>(
    src: &Tile<T>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(bool) {
    let packing = src.packing();
    let shared = src.is_shared();
    let one_level = src.factor_levels();
    let e2m1 = comptime!(matches!(
        packing,
        Packing::Packed {
            field: Field::Quant(QuantValue::E2M1) | Field::ConvertedE2M1
        }
    ));
    let blocks = comptime!(k.is_multiple_of(E2M1_SCALE_BLOCK));
    let offered = offers_block_scaled(m, n, k);
    comptime!(e2m1 && shared && one_level == 1 && blocks && offered)
}

/// Whether both `lhs` and `rhs` contract through the device's block-scaled instruction of
/// `m × n × k` ([`block_scales_here`]): the instruction takes both or neither.
#[cube]
pub(crate) fn contracts_block_scaled<L: Numeric, R: Numeric>(
    lhs: &Tile<L>,
    rhs: &Tile<R>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(bool) {
    let lhs_scaled = block_scales_here(lhs, m, n, k);
    let rhs_scaled = block_scales_here(rhs, m, n, k);
    comptime!(lhs_scaled && rhs_scaled)
}

/// Whether the device offers the block-scaled instruction of `m × n × k` over `e2m1` operands
/// under `e4m3` scales, read off its properties at expansion.
#[cube]
fn offers_block_scaled(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let fp4 = ElemType::Float(FloatKind::E2M1x2);
        let wanted = ScaledMmaConfig {
            a_type: fp4,
            b_type: fp4,
            cd_type: ElemType::Float(FloatKind::F32),
            scales_type: ElemType::Float(FloatKind::E4M3),
            m: m as u32,
            n: n as u32,
            k: k as u32,
            scales_factor: (k / E2M1_SCALE_BLOCK) as u32,
        };
        scope
            .state()
            .device_properties
            .as_ref()
            .is_some_and(|properties| properties.features.matmul.scaled_mma.contains(&wanted))
    })
}

/// Load a block-scaled operand: each unit's words of `src`'s stored `e2m1` values where the
/// instruction places its registers, each a run of eight values along `k`, and the four scales
/// of the row (for `A`) or column (for `B`) its scale register serves, one every
/// [`E2M1_SCALE_BLOCK`] values of the instruction's `k`. `layout` is the window's: an `A` read
/// along its rows, a `B` along its columns where it lies col-major, the stored words running along
/// `k` either way.
///
/// The words come in one `ldmatrix` for the whole fragment where `io` lets the role load through
/// it and a stage in shared memory holds them ([`load_block_scaled_ldmatrix`]); a word at a time
/// otherwise.
///
/// Scales stored as words of four, as an NVFP4 checkpoint packs them, are one load each, straight
/// into the register; scales stored one to a value are read and narrowed one at a time.
#[cube]
fn load_block_scaled<T: Numeric, NV: Size, NS: Size>(
    src: &Tile<T>,
    values: &mut Array<Vector<e2m1x2, NV>>,
    scales: &mut Array<Vector<e4m3, NS>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] shape: (usize, usize, usize),
) {
    let (m, n, k) = comptime!(shape);
    let def = block_scaled_definition::<f32>(m, n, k);
    let space = comptime!(src.place.space.clone());
    let axes = comptime!(MatrixAxes::edges(&space));
    // The window's own `(row, col)` of the matrix's `(row, col)`: a col-major window is the
    // operand's matrix transposed.
    let transposed = comptime!(match layout {
        MatrixLayout::RowMajor => false,
        MatrixLayout::ColMajor => true,
        MatrixLayout::Undefined => {
            panic!("MmaData::load_block_scaled: a block-scaled window is row- or col-major")
        }
    });
    comptime!(assert!(
        (ident == MatrixIdent::A) != transposed,
        "MmaData::load_block_scaled: the instruction reads both operands along `k`, so a \
         block-scaled window holds its words along `k`: an `A` row-major and a `B` col-major"
    ));
    // A register of the instruction is one stored word, eight values; the window may be read a
    // whole line of words at a time, up to the sixteen bytes one unit reads at once.
    let load = src.vector_tile();
    let line_words = comptime!(load.values() / E2M1_PER_WORD);
    comptime!(assert!(
        load.values().is_multiple_of(E2M1_PER_WORD)
            && LDMATRIX_ROW_WORDS.is_multiple_of(line_words),
        "MmaData::load_block_scaled: a register of the instruction is one stored word, eight \
         `e2m1` values, read out of lines of whole words up to sixteen bytes; this window is read \
         {} values a load",
        load.values()
    ));
    let size!(WP) = line_words;
    let words = src.nd_words::<WP>(comptime!(Guard::Checked));
    let unit = UNIT_POS_PLANE;
    let offered = loads_words_as_matrix();
    let shared = src.is_shared();
    let gathered = src.gathered();
    let method = comptime!(match offered && shared && !gathered {
        true => io.load_method(ident),
        false => LoadMethod::Manual,
    });
    match method {
        LoadMethod::LoadMatrix => {
            load_block_scaled_ldmatrix(src, &words, values, &def, ident, transposed, line_words);
        }
        LoadMethod::Manual => {
            // An `e2m1x2` holds two values.
            let per_register = def.vector_size(ident);
            let registers = def.vectors_per_lane(ident);
            #[unroll]
            for i in 0..registers {
                let (row, col) =
                    def.position_of_nth(unit, comptime!((i * per_register * 2) as u32), ident);
                let (window_row, window_col) = if comptime!(transposed) {
                    (col, row)
                } else {
                    (row, col)
                };
                let at =
                    TileMatrix::value_coords(window_row, window_col, 0usize, &space, axes, 1usize);
                // The window's columns run along `k` whichever its layout.
                let line = words.read(load.index(&at, &space));
                let word =
                    window_col / comptime!(E2M1_PER_WORD as u32) % comptime!(line_words as u32);
                values[i] =
                    Vector::<e2m1x2, NV>::reinterpret(line.extract_dynamic(word.cast::<usize>()));
            }
        }
    }
    // The row of `A` or the column of `B` this unit's scale register serves, its scales along
    // `k` one a block.
    let served = def.scales_index(unit, ident);
    let factor = src.innermost_factor();
    let packed_scales = factor.holds_words();
    if comptime!(packed_scales) {
        // The step's four scales as the operand stores them: one word, the register itself.
        comptime!(assert!(
            k / E2M1_SCALE_BLOCK == 4,
            "MmaData::load_block_scaled: a word holds four block scales, and this step reads {}",
            k / E2M1_SCALE_BLOCK
        ));
        let at = TileMatrix::value_coords(served, 0u32.runtime(), 0usize, &space, axes, 1usize);
        let word = factor.word_at(&at, comptime!(space.clone()));
        scales[0] = Vector::<e4m3, NS>::reinterpret(word);
    } else {
        let mut register = Vector::<e4m3, NS>::empty();
        #[unroll]
        for b in 0..comptime!(k / E2M1_SCALE_BLOCK) {
            // The window's rows are the served axis whichever its layout: an `A`'s rows, a
            // `B`'s columns lying col-major.
            let along_k = comptime!((b * E2M1_SCALE_BLOCK) as u32);
            let at =
                TileMatrix::value_coords(served, along_k.runtime(), 0usize, &space, axes, 1usize);
            let scale = factor.at_coords(&at, comptime!(space.clone()));
            register.insert(b, e4m3::cast_from(scale));
        }
        scales[0] = register;
    }
}

/// The words of a block-scaled fragment in one `ldmatrix`: the instruction's registers lie as a
/// 16-bit instruction's do, a register one 8×8 matrix of 16-bit cells, eight rows of four words
/// along `k`, so unit `l` addresses row `l % 8` of matrix `l / 8` and the instruction hands each
/// unit its register's word. The matrices lie where the registers do
/// ([`MmaDefinition::position_of_nth`] of unit 0), so the rows a unit addresses are the rows the
/// word-at-a-time load reads, sixteen bytes at a time, and no matrix is transposed: both
/// operands' windows hold their words along `k`, as the registers run.
///
/// The address is the window's own arrangement of the row ([`Masked::line_slice`]), so a swizzled
/// stage is read where its fill wrote it.
#[cube]
fn load_block_scaled_ldmatrix<T: Numeric, WP: Size, NV: Size>(
    src: &Tile<T>,
    words: &Masked<'_, Vector<u32, WP>, CoordsDyn>,
    values: &mut Array<Vector<e2m1x2, NV>>,
    def: &MmaDefinition<e2m1x2, e2m1x2, f32>,
    #[comptime] ident: MatrixIdent,
    #[comptime] transposed: bool,
    #[comptime] line_words: usize,
) {
    let space = comptime!(src.place.space.clone());
    let axes = comptime!(MatrixAxes::edges(&space));
    let load = src.vector_tile();
    let rank = comptime!(space.rank());
    // An `e2m1x2` holds two values; a register is one word.
    let per_register = def.vector_size(ident);
    let registers = def.vectors_per_lane(ident);
    let unit = UNIT_POS_PLANE;
    let row_in_matrix = unit % 8;
    let nth_matrix = unit / 8 % comptime!(registers as u32);
    let (row, col) =
        def.position_of_nth(0, nth_matrix * comptime!((per_register * 2) as u32), ident);
    // The window's rows are the served axis whichever its layout, its words along `k`.
    let (window_row, window_col) = if comptime!(transposed) {
        (col + row_in_matrix, row)
    } else {
        (row + row_in_matrix, col)
    };
    let at = TileMatrix::value_coords(window_row, window_col, 0usize, &space, axes, 1usize);
    // One row of a matrix: sixteen bytes along the window's innermost axis, in its lines.
    let mut run = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        let extent = comptime!(match p == rank - 1 {
            true => (LDMATRIX_ROW_WORDS / line_words) as u32,
            false => 1u32,
        });
        run.push(extent.runtime());
    }
    let row_slice = words.line_slice(load.index(&at, &space), run);
    let regs = def.load_matrix::<Vector<u32, WP>, Const<1>>(row_slice, ident, registers, false);
    #[unroll]
    for i in 0..registers {
        values[i] = Vector::<e2m1x2, NV>::reinterpret(regs[i].extract(0usize));
    }
}

/// Whether the device offers `ldmatrix`, which moves 16-bit cells: words of `e2m1` values move as
/// pairs of them. Read off its properties at expansion.
#[cube]
fn loads_words_as_matrix() -> comptime_type!(bool) {
    intrinsic!(|scope| {
        scope
            .state()
            .device_properties
            .as_ref()
            .is_some_and(|properties| {
                properties
                    .features
                    .matmul
                    .ldmatrix
                    .contains(&ElemType::Float(FloatKind::F16))
            })
    })
}

/// `acc += lhs · rhs` over two block-scaled operand fragments via
/// `MmaDefinition::execute_scaled`.
#[cube]
pub(crate) fn mma_execute_block_scaled<A: Numeric>(
    lhs: &Array<Vector<e2m1x2, NLB>>,
    lhs_scales: &Array<Vector<e4m3, NSB>>,
    rhs: &Array<Vector<e2m1x2, NRB>>,
    rhs_scales: &Array<Vector<e4m3, NSB>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let def = block_scaled_definition::<A>(m, n, k);
    let out = def.execute_scaled(lhs, rhs, &*acc, lhs_scales[0], rhs_scales[0]);
    let num = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num {
        acc[i] = out[i];
    }
}
