//! The manual-mma encoding of a plane tile ([`MmaData`]) and its fragment↔memory transports.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2, e4m3,
    features::ScaledMmaConfig,
    ir::{ElemType, FloatKind},
    prelude::*,
    quant::scheme::QuantValue,
};

use super::load_matrix::{LDMATRIX_ROW_BYTES, load_ldmatrix};
use crate::*;

// Per-role fragment register widths, bound at allocation to `def.vector_size(role)`.
define_size!(pub NL);
define_size!(pub NR);
define_size!(pub NA);
// A block-scaled fragment's widths: its values' per role, and its scales'.
define_size!(pub NLB);
define_size!(pub NRB);
define_size!(pub NSB);

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
    /// `e4m3` scales of the row this unit's scale register serves.
    LhsBlockScaled(Array<Vector<e2m1x2, NLB>>, Vector<e4m3, NSB>),
    /// A block-scaled `B`, as [`LhsBlockScaled`](Self::LhsBlockScaled) is an `A`.
    RhsBlockScaled(Array<Vector<e2m1x2, NRB>>, Vector<e4m3, NSB>),
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
                Vector::empty(),
            ),
            MatrixIdent::B => MmaFragment::new_RhsBlockScaled(
                Array::new(def.vectors_per_lane(MatrixIdent::B)),
                Vector::empty(),
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
                load_block_scaled(src, values, scales, MatrixIdent::A, layout, (m, n, k))
            }
            MmaFragment::RhsBlockScaled(values, scales) => {
                load_block_scaled(src, values, scales, MatrixIdent::B, layout, (m, n, k))
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
        fragment[i] = fragment[i] * factor;
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
#[cube]
fn load_block_scaled<T: Numeric, NV: Size, NS: Size>(
    src: &Tile<T>,
    values: &mut Array<Vector<e2m1x2, NV>>,
    scales: &mut Vector<e4m3, NS>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
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
    let load = src.vector_tile();
    comptime!(assert!(
        load.values() == 8,
        "MmaData::load_block_scaled: a register of the instruction is one stored word, eight \
         `e2m1` values; this window is read {} values a load",
        load.values()
    ));
    let words = src.nd_words::<Const<1>>(comptime!(Guard::Checked));
    let unit = UNIT_POS_PLANE;
    // An `e2m1x2` holds two values.
    let per_register = def.vector_size(ident);
    let registers = def.vectors_per_lane(ident);
    #[unroll]
    for i in 0..registers {
        let (row, col) = def.position_of_nth(unit, comptime!((i * per_register * 2) as u32), ident);
        let (window_row, window_col) = if comptime!(transposed) {
            (col, row)
        } else {
            (row, col)
        };
        let at = matrix_coords(window_row, window_col, 0usize, &space, axes, 1usize);
        let word = words.read(load.index(&at, &space));
        values[i] = Vector::<e2m1x2, NV>::reinterpret(word.extract(0usize));
    }
    // The row of `A` or the column of `B` this unit's scale register serves, its scales along
    // `k` one a block.
    let served = def.scales_index(unit, ident);
    let factor = src.innermost_factor();
    #[unroll]
    for b in 0..comptime!(k / E2M1_SCALE_BLOCK) {
        // The window's rows are the served axis whichever its layout: an `A`'s rows, a `B`'s
        // columns lying col-major.
        let along_k = comptime!((b * E2M1_SCALE_BLOCK) as u32);
        let at = matrix_coords(served, along_k.runtime(), 0usize, &space, axes, 1usize);
        let scale = factor.at_coords(&at, comptime!(space.clone()));
        scales.insert(b, e4m3::cast_from(scale));
    }
}

/// `acc += lhs · rhs` over two block-scaled operand fragments via
/// `MmaDefinition::execute_scaled`.
#[cube]
pub(crate) fn mma_execute_block_scaled<A: Numeric>(
    lhs: &Array<Vector<e2m1x2, NLB>>,
    lhs_scales: Vector<e4m3, NSB>,
    rhs: &Array<Vector<e2m1x2, NRB>>,
    rhs_scales: Vector<e4m3, NSB>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let def = block_scaled_definition::<A>(m, n, k);
    let out = def.execute_scaled(lhs, rhs, &*acc, lhs_scales, rhs_scales);
    let num = def.vectors_per_lane(MatrixIdent::Accumulator);
    #[unroll]
    for i in 0..num {
        acc[i] = out[i];
    }
}
