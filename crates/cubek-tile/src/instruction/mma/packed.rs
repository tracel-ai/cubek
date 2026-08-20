//! The chunked packed-lhs walk: a packed-u32 lhs contracted in whole stored word lines, decoded
//! in registers.
//!
//! One algorithm over two accumulators and two rhs orientations. The accumulators are the
//! memory-backed leaf's block, seeded from its sink per batch matrix
//! ([`register`](super::register)), and the resident [`RegisterData`](crate::RegisterData) block a
//! promoted accumulator keeps across the whole `K` walk; both meet the walk as an `Array` of cells,
//! so only the walk lives here. The orientations are [`PackedWalk`], which the walk is a table
//! over.

use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantValue},
};

use super::register::Block;
use crate::*;

/// How the rhs meets a packed lhs, and so how a stored word is decoded against it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PackedWalk {
    /// An rhs lined along `N`, the plain orientation: one decoded value and one fma per scalar
    /// `K` step, the accumulator lining along `N` with it.
    Scalar,
    /// An rhs lined along the contraction (`x` bound `[N, K]`, `K` innermost): a whole x line
    /// against a whole decoded group of the word per fma, the accumulator staying scalar cells.
    /// The per-value cost drops from a scalar load + scalar fma to a fraction of a vector load +
    /// vector fma, which is what a kernel bound by per-value instructions (its bytes are
    /// quarter-width) needs.
    Vectorized,
}

/// The shape of one chunked packed-lhs walk: the decode geometry its operands fix, plus the walk
/// their orientations pick. What the loop nest needs beyond the operands and the [`Block`] it
/// folds into.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PackedChunk {
    /// The walk the nest runs.
    pub(crate) walk: PackedWalk,
    /// The lhs's quantization scheme, decoded word by word in registers.
    pub(crate) scheme: QuantScheme,
    /// Values per stored `u32` word.
    pub(crate) pack: usize,
    /// Served values per stored lhs line: the binding width times [`pack`](Self::pack).
    pub(crate) lw: usize,
    /// The rhs's line width.
    pub(crate) vw: usize,
    /// Stored lines the contraction tiles into, one chunk each.
    pub(crate) lines: usize,
}

impl PackedChunk {
    /// The accumulator's line width under this walk: the rhs's own, which the scalar walk lines
    /// the accumulator along with, or scalar cells, the vectorized walk's rhs width riding `K`
    /// rather than `N`.
    pub(crate) fn acc_width(&self) -> usize {
        match self.walk {
            PackedWalk::Scalar => self.vw,
            PackedWalk::Vectorized => 1,
        }
    }

    /// The accumulator cells an `n`-wide output edge covers under this walk.
    pub(crate) fn cells(&self, n: usize) -> usize {
        n / self.acc_width()
    }

    /// Stored words per served line, the wide load the chunk walks.
    pub(crate) fn words(&self) -> usize {
        self.lw / self.pack
    }

    /// Rhs-wide groups per word: how many x lines one word's values cover.
    pub(crate) fn groups(&self) -> usize {
        self.pack / self.vw
    }
}

#[cube]
impl PackedChunk {
    /// The walk a packed `lhs` takes against `rhs` into `out`, and the line arithmetic that
    /// follows from it.
    ///
    /// The 2-D form the walk assumes is asserted rather than re-decided:
    /// [`mma_leaf`](super::mma_leaf) routes a gathered or multi-axis contraction to the N-D nest.
    /// One scale covers a whole line — a served line never straddles a scale block
    /// (`validate_scheme`'s `block % vector_size` rule) — re-asserted here because a hand-built
    /// `QuantTileArgLaunch` reaches the leaf without passing the builder.
    pub(crate) fn new<EL: Numeric, ER: Numeric>(
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
    ) -> comptime_type!(PackedChunk) {
        let lhs_gathered = lhs.gathered();
        let rhs_gathered = rhs.gathered();
        comptime!(assert!(
            !lhs_gathered && !rhs_gathered,
            "packed chunk walk: a gathered operand has no 2-D matrix view; it needs the N-D nest"
        ));
        let contracted = comptime!(Space::contracted(&[&lhs.space, &rhs.space], &out).to_vec());
        comptime!(assert!(
            contracted.len() == 1,
            "packed chunk walk: the 2-D microkernel contracts exactly one axis"
        ));

        let scheme = lhs.quant_scheme();
        let pack = lhs.quant_pack();
        // Served values per stored line, and the rhs's own width.
        let lw = lhs.vector_size();
        let vw = rhs.vector_size();
        let k_axis = comptime!(contracted[0]);
        let kc = comptime!(lhs.space.extent(k_axis));
        comptime!(assert!(
            kc.is_multiple_of(lw),
            "packed chunk walk: the walk moves in whole {lw}-value stored lines, which must tile \
             the {kc}-deep contraction"
        ));
        comptime!(assert!(
            block_edges(scheme, lhs.space.rank())[lhs.space.rank() - 1].is_multiple_of(lw),
            "packed chunk walk: a served line may not straddle two scale blocks"
        ));

        comptime! {
            // Whether the rhs lines along the contraction is the whole decision: lined along `N`
            // it serves one value per `K` step, lined along `K` it serves a group of them.
            let walk = if rhs.space.axis_at(rhs.space.rank() - 1) == k_axis && vw > 1 {
                assert!(
                    pack.is_multiple_of(vw),
                    "packed chunk walk: a {vw}-wide x line must divide a word's {pack} values"
                );
                PackedWalk::Vectorized
            } else {
                PackedWalk::Scalar
            };
            PackedChunk {
                walk,
                scheme,
                pack,
                lw,
                vw,
                lines: kc / lw,
            }
        }
    }
}

/// Contract one packed-lhs chunk of `lhs · rhs` into the block `c`: whole stored word lines, one
/// wide load per line, decoded in registers with the line's one scale folded in per line.
///
/// `A` is the block's line width, `X` the rhs's, `WPL` the words a stored line holds. The
/// accumulator behind `c` is the caller's business: the memory-backed leaf seeds a block per batch
/// matrix and commits it back, while a promoted [`RegisterData`](crate::RegisterData) block *is*
/// the accumulator and stays in registers across the whole `K` walk.
#[cube]
pub(crate) fn packed_chunk_walk<
    E: Numeric,
    EL: Numeric,
    ER: Numeric,
    WPL: Size,
    A: Size,
    X: Size,
>(
    c: &mut Array<Vector<E, A>>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    mat: usize,
    #[comptime] chunk: PackedChunk,
    #[comptime] block: Block,
) {
    match comptime!(chunk.walk) {
        PackedWalk::Scalar => walk_scalar::<E, EL, ER, WPL, A>(c, lhs, rhs, mat, chunk, block),
        PackedWalk::Vectorized => {
            walk_vectorized::<E, EL, ER, WPL, A, X>(c, lhs, rhs, mat, chunk, block)
        }
    }
}

/// [`PackedWalk::Scalar`]: one decode and one fma per scalar `K` step, no served-width vector
/// ever built (the fused view would need one per step, wider than any real vector).
#[cube]
fn walk_scalar<E: Numeric, EL: Numeric, ER: Numeric, WPL: Size, A: Size>(
    c: &mut Array<Vector<E, A>>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    mat: usize,
    #[comptime] chunk: PackedChunk,
    #[comptime] block: Block,
) {
    let (words, scales) = lhs.matrix_packed::<WPL>(mat);
    let rhs_mat = rhs.matrix_transparent::<ER, A, A>(mat);
    let (mr, nr) = comptime!((block.mr, block.nr));
    let unroll = comptime!(block.unroll);
    let (pack, lw, scheme) = comptime!((chunk.pack, chunk.lw, chunk.scheme));
    // The scalar decode is a one-value group.
    let size!(ONE) = 1usize;

    for l in 0..comptime!(chunk.lines) {
        // Each row's stored line and its one scale, held for the whole chunk: one wide load
        // per line, where the fused view issues one per scalar `K` step.
        let mut a_words = Array::<Vector<u32, WPL>>::new(mr);
        let mut a_scales = Array::<E>::new(mr);
        #[unroll(unroll)]
        for i in 0..mr {
            a_words[i] = words.read((i as u32, l as u32));
            a_scales[i] = E::cast_from(scales.read((i as u32, l as u32)));
        }
        #[unroll]
        for w in 0..comptime!(chunk.words()) {
            #[unroll]
            for j in 0..pack {
                let p = (l * lw + comptime!(w * pack + j)) as u32;
                let mut b = Array::<Vector<E, A>>::new(nr);
                #[unroll(unroll)]
                for n in 0..nr {
                    b[n] = Vector::<E, A>::cast_from(rhs_mat.read((p, n as u32)));
                }
                #[unroll(unroll)]
                for i in 0..mr {
                    // Value `j` of word `w`, then the scale — the order the fused view's
                    // `q * scale` reads in.
                    let q = decode_packed::<E, ONE>(a_words[i].extract(w), j, scheme);
                    let a = Vector::<E, A>::cast_from(q.extract(0usize) * a_scales[i]);
                    #[unroll(unroll)]
                    for n in 0..nr {
                        c[i * nr + n] = fma(a, b[n], c[i * nr + n]);
                    }
                }
            }
        }
    }
}

/// [`PackedWalk::Vectorized`]: each fma covers a whole x line against a whole decoded group of
/// the word.
///
/// The line's one scale folds in **once per stored line**: the group partials accumulate unscaled
/// in an `X`-wide register, fold horizontally at the end of the line, and one fma per cell applies
/// the scale — legal because a served line never straddles a scale block.
#[cube]
fn walk_vectorized<E: Numeric, EL: Numeric, ER: Numeric, WPL: Size, A: Size, X: Size>(
    c: &mut Array<Vector<E, A>>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    mat: usize,
    #[comptime] chunk: PackedChunk,
    #[comptime] block: Block,
) {
    let (words, scales) = lhs.matrix_packed::<WPL>(mat);
    let rhs_mat = rhs.matrix_transparent::<ER, X, X>(mat);
    let (mr, nr) = comptime!((block.mr, block.nr));
    let unroll = comptime!(block.unroll);
    let (lw, vw, scheme) = comptime!((chunk.lw, chunk.vw, chunk.scheme));

    for l in 0..comptime!(chunk.lines) {
        let mut a_words = Array::<Vector<u32, WPL>>::new(mr);
        let mut a_scales = Array::<E>::new(mr);
        #[unroll(unroll)]
        for i in 0..mr {
            a_words[i] = words.read((i as u32, l as u32));
            a_scales[i] = E::cast_from(scales.read((i as u32, l as u32)));
        }
        // Unscaled `X`-wide partials for this line, one per block cell.
        let mut q = Array::<Vector<E, X>>::new(comptime!(block.cells()));
        #[unroll(unroll)]
        for i in 0..comptime!(block.cells()) {
            q[i] = Vector::<E, X>::cast_from(E::from_int(0));
        }
        #[unroll]
        for w in 0..comptime!(chunk.words()) {
            #[unroll]
            for g in 0..comptime!(chunk.groups()) {
                // The x line covering values `g·vw .. (g+1)·vw` of word `w`.
                let kline = (l * comptime!(lw / vw) + comptime!(w * chunk.groups() + g)) as u32;
                let mut b = Array::<Vector<E, X>>::new(nr);
                #[unroll(unroll)]
                for n in 0..nr {
                    b[n] = Vector::<E, X>::cast_from(rhs_mat.read((n as u32, kline)));
                }
                #[unroll(unroll)]
                for i in 0..mr {
                    let qv = decode_packed::<E, X>(a_words[i].extract(w), g, scheme);
                    #[unroll(unroll)]
                    for n in 0..nr {
                        q[i * nr + n] = fma(qv, b[n], q[i * nr + n]);
                    }
                }
            }
        }
        // Fold the line: one horizontal sum and one scale fma per cell.
        #[unroll(unroll)]
        for i in 0..mr {
            #[unroll(unroll)]
            for n in 0..nr {
                let partial = horizontal_sum::<E, X>(q[i * nr + n]);
                c[i * nr + n] = fma(
                    Vector::<E, A>::cast_from(a_scales[i]),
                    Vector::<E, A>::cast_from(partial),
                    c[i * nr + n],
                );
            }
        }
    }
}

/// Values `g·W .. (g+1)·W` of a packed `u32` word as a `W`-wide vector, sign-extended by two
/// shifts: left-shift the value's bits to the top, arithmetic-shift back down. A one-wide group
/// is the scalar walk's whole decode.
///
/// Element-for-element what cubecl's `cast_masked` computes for the integer quant values, so the
/// chunked walk and the fused view agree bit-exactly (integer arithmetic, no rounding). The
/// minifloat values live in other stores (`PackedNative`/`Native`) and are refused rather than
/// mis-decoded.
#[cube]
pub(crate) fn decode_packed<E: Numeric, W: Size>(
    word: u32,
    #[comptime] g: usize,
    #[comptime] scheme: QuantScheme,
) -> Vector<E, W> {
    comptime!(assert!(
        matches!(
            scheme.value,
            QuantValue::Q8F
                | QuantValue::Q8S
                | QuantValue::Q4F
                | QuantValue::Q4S
                | QuantValue::Q2F
                | QuantValue::Q2S
        ),
        "packed chunk walk: the decode serves integer quant values only, got {:?}",
        scheme.value
    ));
    let size_bits = comptime!(scheme.size_bits_value());
    let mut out = Vector::<E, W>::empty();
    #[unroll]
    for m in 0..out.vector_size() {
        // Element `m` serves value `g·W + m` of the word, whose bits sit at
        // `size_bits · (g·W + m)`: shift them to the top, arithmetic-shift back.
        let up = 32 - size_bits * (g * out.vector_size() + m + 1);
        let shifted = i32::reinterpret(word << (up as u32)) >> ((32 - size_bits) as i32);
        out.insert(m, E::cast_from(shifted));
    }
    out
}

/// Horizontal sum of a `W`-wide vector into its scalar element.
#[cube]
pub(crate) fn horizontal_sum<E: Numeric, W: Size>(v: Vector<E, W>) -> E {
    let mut total = v.extract(0usize);
    #[unroll]
    for m in 1..v.vector_size() {
        total += v.extract(m);
    }
    total
}
