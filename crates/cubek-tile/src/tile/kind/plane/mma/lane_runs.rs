//! A manual-mma fragment deeper along `k` than its instruction: the registers of several
//! instructions, which a contraction issues one after another into the same accumulator.
//!
//! A sum does not care which value of `k` it meets in which slot of which instruction, only that
//! both factors hold the same `k` there. So a fragment deeper than its instruction lays `k` out
//! in **lane runs**: each unit's slots, across all of the fragment's instructions, hold one
//! contiguous run of the window's `k`, the run of the unit's quad in it. A unit then reads its
//! share of the window in loads as wide as the operand is served, sixteen bytes of 16-bit values
//! or thirty-two packed `e2m1` values, rather than a pair of cells an instruction; and a packed
//! operand decodes and scales a whole word in registers, where the instruction reads it.
//!
//! The order is the depth's consequence, not a choice: a fragment one instruction deep keeps the
//! instruction's own order, and every contraction that names its instruction's depth is the one
//! it was. Both factors of one contraction are the same depth, so they agree on the order.
//!
//! The run a slot belongs to is read off the 16-bit `m16n8k16` layout, which every unit's
//! registers follow the same way: unit `u` holds the cells of quad `u % 4` along `k`, a pair of
//! adjacent cells a register, its first pair from the instruction's first eight values of `k` and
//! its second from the last eight. [`lane_runs_here`] admits the targets whose layout this is,
//! and the component test checks the map against `position_of_nth` on every unit.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2,
    ir::types::Fp8Format,
    prelude::*,
    quant::scheme::QuantValue,
};

use super::widths::*;
use crate::*;

/// Slots of one instruction's `k` a unit holds, on the 16-bit `m16n8k16` layout: four of sixteen.
pub(crate) const SLOTS_PER_UNIT: usize = 4;

/// Adjacent slots one register holds: a pair of 16-bit values.
const SLOTS_PER_REGISTER: usize = 2;

/// Units sharing one row of `A` or column of `B`, each holding its own quad of `k`.
const QUADS: u32 = 4;

/// `e2m1` values one stored word holds.
const E2M1_PER_WORD: usize = 8;

/// Values one `e2m1` block scale covers along `k`.
const E2M1_BLOCK: usize = 16;

/// The depth along `k` one instruction of an `m × n` fragment of `T` contracts: `k` itself where
/// the device offers that instruction, else the deepest one it offers that divides `k`, else `k`,
/// which the expansion of the instruction then refuses as it always has.
// `T` is read by the expansion, which is where a type has its element.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
pub(crate) fn instruction_depth<T: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(usize) {
    intrinsic!(|scope| {
        let element = T::elem_type(scope);
        let Some(properties) = scope.state().device_properties.as_ref() else {
            return k;
        };
        let depths = properties
            .features
            .matmul
            .mma
            .iter()
            .filter(|config| {
                config.a_type == element
                    && config.b_type == element
                    && (config.m as usize, config.n as usize) == (m, n)
            })
            .map(|config| config.k as usize)
            .collect::<Vec<_>>();
        match depths.contains(&k) {
            true => k,
            false => depths
                .into_iter()
                .filter(|&depth| k.is_multiple_of(depth))
                .max()
                .unwrap_or(k),
        }
    })
}

/// Instructions one `m × n × k` fragment of `T` holds along `k`: one, unless it is deeper than its
/// instruction, when it holds `k` in lane runs.
#[cube]
pub(crate) fn lane_run_steps<T: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) -> comptime_type!(usize) {
    let depth = instruction_depth::<T>(m, n, k);
    comptime!(k / depth)
}

/// Whether the target lays a 16-bit `m16n8k16` instruction's registers out as the lane-run map
/// reads them: a 32-unit plane, `A` along its rows and `B` along its columns, every cell held
/// once. A target that agrees on these and lays the slots otherwise fails the component test.
#[cube]
fn lays_out_lane_runs() -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let mma = &scope.state().target_properties.mma;
        mma.const_plane_size == 32
            && mma.register_layout_a == MatrixLayout::RowMajor
            && mma.register_layout_b == MatrixLayout::ColMajor
            && mma.register_duplication_a == 1
            && mma.register_duplication_b == 1
    })
}

/// Why `src`, a window of an `m × n × k` fragment of `T` in role `ident`, cannot load in lane
/// runs; `None` where it can.
///
/// It can where the fragment is deeper than its instruction, a 16-bit `m16n8k16` one, on a target
/// [`lays_out_lane_runs`]; its window runs along `k` (an `A` row-major, a `B` col-major); and it
/// is plain, or `e2m1` codes the device converts under one level of scales a value times its
/// scale is exact under in 16 bits, NVFP4's `e4m3` or a power of two. A coarser level is the
/// product's to apply to the accumulator, as it is for the block-scaled instruction.
#[cube]
pub(crate) fn lane_runs_refusal<T: Numeric>(
    src: &Tile<T>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] shape: (usize, usize, usize),
) -> comptime_type!(Option<&'static str>) {
    let (m, n, k) = comptime!(shape);
    let depth = instruction_depth::<T>(m, n, k);
    let laid_out = lays_out_lane_runs();
    let bytes = T::size().comptime();
    let packing = src.packing();
    let levels = src.factor_levels();
    let stored = src.innermost_factor().inner_stored();
    let along_k = comptime!(match ident {
        MatrixIdent::A => layout == MatrixLayout::RowMajor,
        MatrixIdent::B => layout == MatrixLayout::ColMajor,
        MatrixIdent::Accumulator => false,
    });
    let e2m1 = comptime!(matches!(
        packing,
        Packing::Packed {
            field: Field::Quant(QuantValue::E2M1) | Field::ConvertedE2M1
        }
    ));
    let exact_scales = comptime!(matches!(
        stored,
        Some(Packing::Packed {
            field: Field::Fp8(Fp8Format::E4M3 | Fp8Format::UE8M0)
        })
    ));
    comptime!(if k <= depth {
        Some("the fragment is one instruction deep, which keeps the instruction's own order")
    } else if (m, n, depth) != (16, 8, 16) || bytes != 2 || !laid_out {
        Some("lane runs read the 16-bit m16n8k16 layout, which this target or element is not")
    } else if !along_k {
        Some("a window loads in lane runs along `k`: an A row-major, a B col-major")
    } else if packing == Packing::Plain && levels == 0 {
        None
    } else if !e2m1 {
        Some("a packed or scaled window loads in lane runs only as e2m1 codes the device converts")
    } else if levels != 1 || !exact_scales {
        Some(
            "e2m1 codes load in lane runs under one level of e4m3 or power-of-two scales, whose \
             product with a code is exact in 16 bits",
        )
    } else {
        None
    })
}

/// Whether `src` loads in lane runs ([`lane_runs_refusal`]).
#[cube]
pub(crate) fn lane_runs_here<T: Numeric>(
    src: &Tile<T>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] shape: (usize, usize, usize),
) -> comptime_type!(bool) {
    let refusal = lane_runs_refusal::<T>(src, ident, layout, shape);
    comptime!(refusal.is_none())
}

/// Rows of the window `space` a fragment `k` deep loads from: its free edge, which a product of
/// fewer rows than the fragment's leaves short of them, the rest read as zeros by the checked
/// view.
fn window_rows(space: &Space, k: usize) -> usize {
    space
        .axes()
        .map(|axis| space.extent(axis))
        .product::<usize>()
        / k
}

/// The register of one instruction of role `ident` that holds pair `pair` of a unit's slots, in
/// half `half` of its rows: an `A` holds rows `g` and `g + 8` in alternate registers, its first
/// pair of `k` in registers 0 and 1; a `B` holds one column, a pair a register.
fn register_of(ident: MatrixIdent, pair: usize, half: usize) -> usize {
    match ident {
        MatrixIdent::A => pair * 2 + half,
        MatrixIdent::B => pair,
        MatrixIdent::Accumulator => panic!("lane runs: an accumulator holds no slots of `k`"),
    }
}

/// Halves of the rows one unit holds: an `A`'s rows `g` and `g + 8`, a `B`'s one column.
fn halves_of(ident: MatrixIdent) -> usize {
    match ident {
        MatrixIdent::A => 2,
        MatrixIdent::B => 1,
        MatrixIdent::Accumulator => panic!("lane runs: an accumulator holds no slots of `k`"),
    }
}

/// Load `fragment`, `steps` instructions of role `ident` deep, from `src` in lane runs: each unit
/// reads the run of its quad along each of its rows (an `A`) or its column (a `B`), and places
/// slot `j` of instruction `s` at offset `4s + j` of it.
#[cube]
pub(crate) fn load_lane_runs<T: Numeric, N: Size>(
    src: &Tile<T>,
    fragment: &mut Array<Vector<T, N>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] shape: (usize, usize, usize),
) {
    let refusal = lane_runs_refusal::<T>(src, ident, layout, shape);
    comptime!(if let Some(why) = refusal {
        panic!("MmaData::load: a fragment deeper than its instruction loads in lane runs: {why}")
    });
    let (m, n, k) = comptime!(shape);
    let depth = instruction_depth::<T>(m, n, k);
    let steps = comptime!(k / depth);
    let run = comptime!(SLOTS_PER_UNIT * steps);
    let def = MmaDefinition::<T, T, T>::new(m, n, depth);
    let per_step = def.vectors_per_lane(ident);
    let unit = UNIT_POS_PLANE;
    // The unit's row of `A` or column of `B` within a half, and its quad's run of `k`.
    let group = unit / QUADS;
    let first = (unit % QUADS) * comptime!(run as u32);
    let packing = src.packing();
    match comptime!(packing) {
        Packing::Plain => {
            load_plain_runs::<T, N>(src, fragment, ident, (m, n, k), group, first, run, per_step)
        }
        Packing::Packed { .. } => {
            load_e2m1_runs::<T, N>(src, fragment, ident, (m, n, k), group, first, run, per_step)
        }
    }
}

/// [`load_lane_runs`] of a plain window: each run read a served line at a time.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_plain_runs<T: Numeric, N: Size>(
    src: &Tile<T>,
    fragment: &mut Array<Vector<T, N>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] shape: (usize, usize, usize),
    group: u32,
    first: u32,
    #[comptime] run: usize,
    #[comptime] per_step: usize,
) {
    let (m, n, k) = comptime!(shape);
    let width = src.vector_size();
    comptime!(assert!(
        run.is_multiple_of(width),
        "MmaData::load: a {width}-wide line runs past a unit's {run}-cell run of `k`; serve the \
         operand at most {run} wide"
    ));
    let size!(W) = width;
    let _ = (m, n);
    let rows = comptime!(window_rows(&src.place.space, k));
    // The window read along its rows, `k` running along its lines.
    let view = src.fragment_matrix_packed::<W>(rows, k);
    let lines = comptime!(run / width);
    #[unroll]
    for half in 0..comptime!(halves_of(ident)) {
        let row = group + comptime!((half * 8) as u32);
        let mut cells = Array::<Vector<T, W>>::new(lines);
        #[unroll]
        for l in 0..lines {
            cells[l] = view.read((row, first / comptime!(width as u32) + comptime!(l as u32)));
        }
        #[unroll]
        for offset in 0..comptime!(run / SLOTS_PER_REGISTER) {
            // Cells `2 · offset` and the next of the run: instruction `s`, pair `pair`.
            let cell = comptime!(offset * SLOTS_PER_REGISTER);
            let (s, pair) = comptime!((cell / SLOTS_PER_UNIT, cell % SLOTS_PER_UNIT / 2));
            let register = comptime!(s * per_step + register_of(ident, pair, half));
            let mut vector = Vector::<T, N>::empty();
            #[unroll]
            for e in 0..SLOTS_PER_REGISTER {
                let at = comptime!(cell + e);
                vector.insert(
                    e,
                    cells[comptime!(at / width)].extract(comptime!(at % width)),
                );
            }
            fragment[register] = vector;
        }
    }
}

/// [`load_lane_runs`] of `e2m1` codes under one level of block scales: each run read as the
/// stored words it is, a load at a time, each word decoded to eight values and multiplied by its
/// block's scale in `T`, exact under [`lane_runs_refusal`]'s conditions.
#[cube]
#[allow(clippy::too_many_arguments)]
fn load_e2m1_runs<T: Numeric, N: Size>(
    src: &Tile<T>,
    fragment: &mut Array<Vector<T, N>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] shape: (usize, usize, usize),
    group: u32,
    first: u32,
    #[comptime] run: usize,
    #[comptime] per_step: usize,
) {
    let (_m, _n, k) = comptime!(shape);
    let space = comptime!(src.place.space.clone());
    // The window's rows are the role's free axis and its columns every digit of `k`, however
    // many axes spell it.
    let rows = comptime!(window_rows(&space, k));
    let axes = comptime!(MatrixAxes::new(&space, rows, k).unwrap_or_else(|no| {
        panic!("MmaData::load: a lane-run window is a {rows} x {k} matrix: {no:?}")
    }));
    let load = src.vector_tile();
    let values = comptime!(load.values());
    // A load lies along `k` alone, its words in `k`'s order: a stored tile whose load also
    // spans columns would place another column's codes in this unit's run.
    let along_k = comptime!(
        load.space()
            .axes()
            .all(|axis| (axes.col_split..space.rank()).any(|p| space.axis_at(p) == axis))
    );
    comptime!(assert!(
        along_k,
        "MmaData::load: a unit's run of `k` reads loads along `k` alone, and this window's load \
         spans {:?}",
        load.space()
    ));
    comptime!(assert!(
        values.is_multiple_of(E2M1_PER_WORD) && run.is_multiple_of(values),
        "MmaData::load: a unit's {run}-value run of `k` reads whole loads of e2m1 words, and this \
         window is read {values} values a load"
    ));
    comptime!(assert!(
        run.is_multiple_of(E2M1_BLOCK) && k.is_multiple_of(E2M1_BLOCK),
        "MmaData::load: a unit's {run}-value run of `k` holds whole blocks of sixteen, one scale \
         each"
    ));
    let size!(WP) = comptime!(values / E2M1_PER_WORD);
    let words = src.nd_words::<WP>(comptime!(Guard::Checked));
    let factor = src.innermost_factor();
    #[unroll]
    for half in 0..comptime!(halves_of(ident)) {
        // The window's rows are the role's free axis: an `A`'s rows, a `B`'s columns.
        let row = group + comptime!((half * 8) as u32);
        let mut scales = Array::<T>::new(comptime!(run / E2M1_BLOCK));
        #[unroll]
        for b in 0..comptime!(run / E2M1_BLOCK) {
            let at = TileMatrix::value_coords(
                row,
                first + comptime!((b * E2M1_BLOCK) as u32),
                0usize,
                &space,
                axes,
                1usize,
            );
            scales[b] = T::cast_from(factor.at_coords(&at, comptime!(space.clone())));
        }
        #[unroll]
        for l in 0..comptime!(run / values) {
            let at = TileMatrix::value_coords(
                row,
                first + comptime!((l * values) as u32),
                0usize,
                &space,
                axes,
                1usize,
            );
            let line = words.read(load.index(&at, &space));
            #[unroll]
            for w in 0..comptime!(values / E2M1_PER_WORD) {
                let word_start = comptime!(l * values + w * E2M1_PER_WORD);
                let codes = Vector::<e2m1x2, Const<4>>::reinterpret(line.extract(w));
                let decoded = Vector::<T, Const<8>>::cast_from(codes)
                    * Vector::<T, Const<8>>::new(scales[comptime!(word_start / E2M1_BLOCK)]);
                #[unroll]
                for p in 0..comptime!(E2M1_PER_WORD / SLOTS_PER_REGISTER) {
                    let cell = comptime!(word_start + p * SLOTS_PER_REGISTER);
                    let (s, pair) = comptime!((cell / SLOTS_PER_UNIT, cell % SLOTS_PER_UNIT / 2));
                    let register = comptime!(s * per_step + register_of(ident, pair, half));
                    let mut vector = Vector::<T, N>::empty();
                    #[unroll]
                    for e in 0..SLOTS_PER_REGISTER {
                        vector.insert(e, decoded.extract(comptime!(p * SLOTS_PER_REGISTER + e)));
                    }
                    fragment[register] = vector;
                }
            }
        }
    }
}

/// `acc += lhs · rhs` over fragments `steps` instructions deep, each `m × n × depth`, issued in
/// turn into the one accumulator.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mma_execute_lane_runs<L: Numeric, R: Numeric, A: Numeric>(
    lhs: &Array<Vector<L, NL>>,
    rhs: &Array<Vector<R, NR>>,
    acc: &mut Array<Vector<A, NA>>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] depth: usize,
    #[comptime] steps: usize,
) {
    let def = MmaDefinition::<L, R, A>::new(m, n, depth);
    let a_registers = def.vectors_per_lane(MatrixIdent::A);
    let b_registers = def.vectors_per_lane(MatrixIdent::B);
    #[unroll]
    for s in 0..steps {
        let mut a = Array::<Vector<L, NL>>::new(a_registers);
        let mut b = Array::<Vector<R, NR>>::new(b_registers);
        #[unroll]
        for i in 0..a_registers {
            a[i] = lhs[comptime!(s * a_registers + i)];
        }
        #[unroll]
        for i in 0..b_registers {
            b[i] = rhs[comptime!(s * b_registers + i)];
        }
        def.execute_inplace(&a, &b, acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit's slots of one instruction land at offsets `4s + j` of its run, in the order the
    /// registers hold them: an `A`'s two pairs in registers 0/2 (rows `g`) and 1/3 (rows `g + 8`).
    #[test]
    fn a_registers_pairs_alternate_their_rows() {
        let a: Vec<_> = (0..2)
            .flat_map(|pair| (0..2).map(move |half| register_of(MatrixIdent::A, pair, half)))
            .collect();
        assert_eq!(a, vec![0, 1, 2, 3]);
        assert_eq!(register_of(MatrixIdent::B, 1, 0), 1);
    }
}
