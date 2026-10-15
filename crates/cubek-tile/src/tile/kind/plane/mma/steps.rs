//! A manual-mma fragment deeper along `k` than its instruction, loaded out of shared memory a step
//! at a time in the instruction's own register order.
//!
//! [Lane runs](super::lane_runs) reorder `k` so a unit reads its share in whole loads, which pays
//! where the window lies in global memory. Out of shared memory a step loads as a fragment one
//! instruction deep does, through `ldmatrix` where the device offers it, so the deeper fragment
//! keeps that transport: each step's registers are the instruction's own over its sixteen of `k`.
//!
//! A window of `e2m1` codes decodes as each step loads: under the 16-bit `m16n8k16` layout a unit
//! holds a column's pairs `2q, 2q + 1` and `2q + 8, 2q + 9` of a step's `k`, which are byte `q` of
//! the column's two words of codes, one block of them under one scale. Words of four `e4m3` block
//! scales give the step its field: a fragment a whole number of words deep starts on a word, so
//! step `s` reads field `s % 4`.

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout, MmaDefinition},
    e2m1x2, e4m3,
    ir::types::Fp8Format,
    prelude::*,
    quant::scheme::QuantValue,
};

use super::block_scaled::{E2M1_PER_WORD, E2M1_SCALE_BLOCK};
use super::lane_runs::{instruction_depth, lays_out_lane_runs};
use crate::*;

/// `e4m3` block scales one word of them holds.
const SCALES_PER_WORD: usize = 4;

/// Codes of `k` one byte holds: a pair, the two cells one 16-bit register holds.
const CODES_PER_BYTE: usize = 2;

/// Why `src`, a shared-memory window of an `m × n × k` fragment of `T` in role `ident`, cannot load
/// a step at a time; `None` where it can.
///
/// It can where the fragment is deeper than its instruction, a 16-bit `m16n8k16` one, on a target
/// that [`lays_out_lane_runs`] (whose layout says which cells a unit holds); and it is plain, or
/// `e2m1` codes along `k` under one level of scales whose product with a code is exact in 16 bits:
/// NVFP4's `e4m3`, as numbers or as the words of four the block-scaled instruction reads, a window
/// under words a whole number of them deep so each step starts its block in its word. A coarser
/// level is the product's to apply to the accumulator.
#[cube]
pub(crate) fn steps_refusal<T: Numeric>(
    src: &Tile,
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
    let factor = src.innermost_factor();
    let stored = factor.inner_stored();
    let words = factor.holds_words();
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
    let exact_scales = comptime!(
        words
            || matches!(
                stored,
                Some(Packing::Packed {
                    field: Field::Fp8(Fp8Format::E4M3 | Fp8Format::UE8M0)
                })
            )
    );
    comptime!(if k <= depth {
        Some("the fragment is one instruction deep, which loads as one")
    } else if (m, n, depth) != (16, 8, 16) || bytes != 2 || !laid_out {
        Some(
            "a step's cells are read off the 16-bit m16n8k16 layout, which this target or element is not",
        )
    } else if packing == Packing::Plain && levels == 0 {
        None
    } else if !e2m1 || !along_k {
        Some("a packed or scaled window loads a step at a time only as e2m1 codes along `k`")
    } else if levels != 1 || !exact_scales {
        Some(
            "e2m1 codes load a step at a time under one level of e4m3 or power-of-two scales, \
             whose product with a code is exact in 16 bits",
        )
    } else if words && !k.is_multiple_of(E2M1_SCALE_BLOCK * SCALES_PER_WORD) {
        Some(
            "a window under words of e4m3 block scales loads a step at a time where it is a \
             whole number of words deep, so each step starts its block in its word",
        )
    } else {
        None
    })
}

/// Fill `fragment`, `steps` of `ident`'s registers one after another, from `src`, a shared-memory
/// window `shape` deep: each step's registers as a fragment of the instruction loads them, over
/// its own sixteen of `k`.
#[cube]
pub(crate) fn load_in_steps<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    #[comptime] ident: MatrixIdent,
    #[comptime] layout: MatrixLayout,
    #[comptime] io: MmaIo,
    #[comptime] shape: (usize, usize, usize),
) {
    let refusal = steps_refusal::<T>(src, ident, layout, shape);
    comptime!(if let Some(why) = refusal {
        panic!(
            "MmaData::load: a fragment deeper than its instruction loads a step at a time: {why}"
        )
    });
    let (m, n, k) = comptime!(shape);
    let depth = instruction_depth::<T>(m, n, k);
    let def = MmaDefinition::<T, T, T>::new(m, n, depth);
    let per_step = def.vectors_per_lane(ident);
    let edges = comptime!(match ident {
        MatrixIdent::A => (m, k),
        MatrixIdent::B => (k, n),
        MatrixIdent::Accumulator => panic!("MmaData::load: an accumulator holds no steps of `k`"),
    });
    let packing = src.packing();
    match comptime!(packing) {
        Packing::Plain =>
        {
            #[unroll]
            for s in 0..comptime!(k / depth) {
                let mut step = Array::<Vector<T, N>>::new(per_step);
                load_fragment(
                    src,
                    &mut step,
                    &def,
                    ident,
                    layout,
                    io,
                    edges,
                    comptime!(s * depth),
                );
                #[unroll]
                for i in 0..per_step {
                    fragment[comptime!(s * per_step + i)] = step[i];
                }
            }
        }
        Packing::Packed { .. } => {
            load_e2m1_steps::<T, N>(src, fragment, &def, ident, comptime!(k / depth), edges)
        }
    }
}

/// Every step of an `e2m1` window under one level of block scales, decoded into the
/// instruction's registers, `steps` of them one after another. Under the 16-bit `m16n8k16` layout
/// a unit holds one row of the window (an `A`'s row, a `B`'s column) per register, its register
/// `i` of step `s` the pair of cells at `2q + 8i + 16s` of `k`, `q` its place in its quad: byte
/// `q` of word `2s + i` of that row's codes. So the unit reads its row's codes for the whole
/// window once, a line at a time, and each block's scale once, and decodes a byte a register.
#[cube]
fn load_e2m1_steps<T: Numeric, N: Size>(
    src: &Tile,
    fragment: &mut Array<Vector<T, N>>,
    def: &MmaDefinition<T, T, T>,
    #[comptime] ident: MatrixIdent,
    #[comptime] steps: usize,
    #[comptime] edges: (usize, usize),
) {
    let space = comptime!(src.place.space.clone());
    // The window's rows are the served axis: an `A`'s rows, a `B`'s columns lying col-major.
    let (rows, k) = comptime!(match ident {
        MatrixIdent::B => (edges.1, edges.0),
        MatrixIdent::A | MatrixIdent::Accumulator => edges,
    });
    let axes = comptime!(MatrixAxes::new(&space, rows, k).unwrap_or_else(|no| {
        panic!("MmaData::load: a stepped window is a {rows} x {k} matrix: {no:?}")
    }));
    let load = src.vector_tile();
    let values = comptime!(load.values());
    comptime!(assert!(
        values.is_multiple_of(E2M1_PER_WORD) && k.is_multiple_of(values),
        "MmaData::load: a stepped e2m1 window is read in whole words, whole lines along its \
         {k} of `k`, and this one {values} values a load"
    ));
    let per_line = comptime!(values / E2M1_PER_WORD);
    let size!(WP) = per_line;
    let words = src.nd_words::<WP>(comptime!(Guard::Checked));
    let factor = src.innermost_factor();
    let words_of_scales = factor.holds_words();
    let vector_size = def.vector_size(ident);
    let per_step = def.vectors_per_lane(ident);
    comptime!(assert!(
        vector_size == CODES_PER_BYTE && per_step == 2,
        "MmaData::load: a stepped e2m1 operand holds two registers a step, a pair of cells each"
    ));
    // The row this unit serves, and its byte of every word: where its quad places it along `k`.
    let (row, col) = def.position_of_nth(UNIT_POS_PLANE, 0u32, ident);
    let (served, first) = match comptime!(ident) {
        MatrixIdent::B => (col, row),
        MatrixIdent::A | MatrixIdent::Accumulator => (row, col),
    };
    // The unit's byte of a word, as a shift: a runtime index into a vector of registers lowers to
    // local memory, a shift stays in registers.
    let shift = first / comptime!(CODES_PER_BYTE as u32) * 8u32;
    // The row's codes over the whole window, a line at a time.
    let lines = comptime!(k / values);
    let mut held = Array::<Vector<u32, WP>>::new(lines);
    #[unroll]
    for l in 0..lines {
        let at = TileMatrix::value_coords(
            served,
            comptime!((l * values) as u32),
            0usize,
            &space,
            axes,
            1usize,
        );
        held[l] = words.read(load.index(&at, &space));
    }
    #[unroll]
    for s in 0..steps {
        let block = TileMatrix::value_coords(
            served,
            comptime!((s * E2M1_SCALE_BLOCK) as u32),
            0usize,
            &space,
            axes,
            1usize,
        );
        let scale = if comptime!(words_of_scales) {
            let fields = Vector::<e4m3, Const<4>>::reinterpret(
                factor.word_at(&block, comptime!(space.clone())),
            );
            T::cast_from(fields.extract(comptime!(s % SCALES_PER_WORD)))
        } else {
            T::cast_from(factor.at_coords(&block, comptime!(space.clone())))
        };
        #[unroll]
        for i in 0..2usize {
            let word = comptime!(2 * s + i);
            let codes = held[comptime!(word / per_line)].extract(comptime!(word % per_line));
            let pair = Vector::<e2m1x2, Const<4>>::reinterpret(codes >> shift).extract(0usize);
            let decoded = Vector::<T, Const<2>>::cast_from(Vector::<e2m1x2, Const<1>>::new(pair))
                * Vector::<T, Const<2>>::new(scale);
            let mut cells = Vector::<T, N>::empty();
            #[unroll]
            for e in 0..CODES_PER_BYTE {
                cells.insert(e, decoded.extract(e));
            }
            fragment[comptime!(s * per_step + i)] = cells;
        }
    }
}
