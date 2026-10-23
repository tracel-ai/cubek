//! The packed KV cache's format as the attention reads it: the widths its codes are stored at,
//! whether its keys carry the QJL fix-up, and the decode of a run of its codes.

use cubecl::prelude::*;

/// The per-coordinate widths a packed cache's codes are stored at: `32 / bits` coordinates share
/// each `u32` word. Keys and values need not agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvBitWidths {
    /// Per-coordinate width of the packed keys.
    pub key: u32,
    /// Per-coordinate width of the packed values.
    pub value: u32,
}

/// Whether a cached vector spends one bit per coordinate on the QJL sign fix-up of its rounding
/// residual.
///
/// Two different algorithms rather than a flag: [`Qjl`](Self::Qjl) emits three extra `O(d²)`
/// passes, writes a second scalar buffer, and shifts every packed index up by one bit. Naming them
/// keeps the choice legible at a call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyFixup {
    /// Round-only (Q_mse): the packed field is entirely the centroid index.
    RoundOnly,
    /// Q_prod: the field's low bit is the sign of the residual's projection, and the residual's
    /// length is stored beside the vector's own.
    Qjl,
}

impl KeyFixup {
    /// Bits per coordinate the fix-up spends, `0` or `1`. The rounding index gets whatever the
    /// packed field has left.
    pub fn bits(self) -> u32 {
        match self {
            KeyFixup::RoundOnly => 0,
            KeyFixup::Qjl => 1,
        }
    }
}

/// A packed cache as a kernel is handed it: the keys' and the values' words, norms and codebooks,
/// the keys' fix-up, and what reading them compiles to. Each operand's vectors hold `coords`
/// coordinates, each a `bits`-wide index into the operand's codebook, packed low first into the
/// u32 words `[b, kv, s, w]`, the vector scaled by its entry of the norms `[b, kv, s, 1]`.
#[derive(CubeType, CubeLaunch)]
pub struct PackedCacheArg {
    pub key_words: OwnedTensor<u32>,
    pub key_norms: OwnedTensor<f32>,
    pub key_codebook: OwnedTensor<f32>,
    /// Without a fix-up, any tensor: a kernel's arguments are fixed, and these are never read.
    pub key_errs: OwnedTensor<f32>,
    pub key_sign_codebook: OwnedTensor<f32>,
    pub value_words: OwnedTensor<u32>,
    pub value_norms: OwnedTensor<f32>,
    pub value_codebook: OwnedTensor<f32>,
    #[cube(comptime)]
    pub coords: usize,
    #[cube(comptime)]
    pub key_bits: u32,
    #[cube(comptime)]
    pub value_bits: u32,
    /// The fix-up the keys carry.
    #[cube(comptime)]
    pub fixup: KeyFixup,
}

/// The `N` consecutive coordinates from `first_coord` of the packed row starting at word
/// `word_base` of `words`, each `bits`-wide field looked up in `table`.
///
/// `N` and the fields a word packs are both powers of two, so a vector's coordinates never
/// straddle a word: the vector sits inside one word or spans whole ones, and each word it covers
/// is loaded once.
#[cube]
pub fn packed_fields<F: Float, N: Size>(
    words: &Tensor<u32>,
    word_base: usize,
    first_coord: usize,
    table: &Tensor<F>,
    #[comptime] bits: u32,
) -> Vector<F, N> {
    let n = N::value();
    let pack_factor = comptime!(32u32 / bits) as usize;
    let mask = comptime!((1u32 << bits) - 1u32);
    let fields_per_word = comptime!(core::cmp::min(n, pack_factor));
    let words_per_vector = comptime!(n / core::cmp::min(n, pack_factor));
    // Zero whenever a vector covers whole words; the in-word start otherwise.
    let field_offset = first_coord % pack_factor;

    let mut out = Vector::<F, N>::zeroed();
    #[unroll]
    for w in 0..words_per_vector {
        let word = words[word_base + first_coord / pack_factor + w];
        #[unroll]
        for j in 0..fields_per_word {
            let shift = (field_offset + j) as u32 * bits;
            let idx = (word >> shift) & mask;
            out.insert(w * fields_per_word + j, table[idx as usize]);
        }
    }
    out
}
