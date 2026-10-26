//! A packed cache as the attention reads it: each operand's codebook indices packed into words,
//! its per-vector norms and the codebook, and the keys' QJL fix-up where there is one.

use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::tile::kind::Field;
use cubecl::tile::{Axis, Launcher, TileArgLaunch, TileSpec};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{B, D, DV, KV, S};

/// A packed cache: both operands at their bit widths, and the keys' fix-up where they carry one.
///
/// Each buffer is a `T`: a caller states the cache in its own tensors and converts it to the
/// default, [`PlainTensor`], which is what a kernel reads: `u32` words whose fields of a row are
/// adjacent, and every norm and codebook in `f32`, the sum's element ([`new`](Self::new)).
#[derive(Debug, Clone)]
pub struct PackedAttentionCache<T = PlainTensor> {
    pub key: PackedAttentionOperand<T>,
    pub value: PackedAttentionOperand<T>,
    pub bits: KvBitWidths,
    pub fixup: Option<QjlFixup<T>>,
}

/// The keys or the values of a packed cache: `words` `[b, kv, s, w]`, each vector's coordinates
/// packed low first as `bits`-wide indices into `codebook`, and the vector scaled by its entry of
/// `norms` `[b, kv, s, 1]`.
#[derive(Debug, Clone)]
pub struct PackedAttentionOperand<T = PlainTensor> {
    pub words: T,
    pub norms: T,
    pub codebook: T,
}

/// The keys' QJL fix-up: each key's residual length `errs` `[b, kv, s, 1]`, and `sign_codebook`,
/// the codebook its indices read the residual's projected sign through.
#[derive(Debug, Clone)]
pub struct QjlFixup<T = PlainTensor> {
    pub errs: T,
    pub sign_codebook: T,
}

/// What a launch compiles about a packed cache: each operand's bit width, and whether the keys
/// carry the QJL fix-up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedLayout {
    pub bits: KvBitWidths,
    pub fixup: KeyFixup,
}

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

impl PackedAttentionCache {
    /// A packed cache of `key` and `value` at `bits`, the keys carrying `fixup` where there is one.
    ///
    /// # Panics
    ///
    /// Words that are not `u32` or whose fields of a row are not adjacent, a norm, codebook or
    /// fix-up table not in `f32`, or a codebook not one-dim with an entry for every index: what a
    /// caller converts before handing the cache over.
    pub fn new(
        key: PackedAttentionOperand,
        value: PackedAttentionOperand,
        bits: KvBitWidths,
        fixup: Option<QjlFixup>,
    ) -> Self {
        const WORD: ElemType = ElemType::UInt(UIntKind::U32);
        const TABLE: ElemType = ElemType::Float(FloatKind::F32);
        let indexes = |table: &PlainTensor, bits: u32| {
            let shape = table.shape();
            assert!(
                shape.len() == 1 && shape[0] >= 1 << bits,
                "a codebook is one-dim, an entry for each of the {} indices: got {shape:?}",
                1u32 << bits
            );
        };
        for (operand, bits) in [(&key, bits.key), (&value, bits.value)] {
            indexes(&operand.codebook, bits);
            assert_eq!(operand.words.elem, WORD, "a packed operand's words are u32");
            assert_eq!(
                operand.words.binding.strides[3], 1,
                "a packed operand's fields of a row are adjacent"
            );
            assert!(
                operand.norms.elem == TABLE && operand.codebook.elem == TABLE,
                "a packed operand's norms and codebook are f32"
            );
        }
        if let Some(fixup) = &fixup {
            indexes(&fixup.sign_codebook, bits.key);
            assert!(
                fixup.errs.elem == TABLE && fixup.sign_codebook.elem == TABLE,
                "the QJL fix-up's lengths and sign codebook are f32"
            );
        }
        Self {
            key,
            value,
            bits,
            fixup,
        }
    }

    /// Coordinates a vector holds: the head dim of the codes.
    pub fn coords(&self) -> usize {
        self.key.words.shape()[3] * 32 / self.bits.key as usize
    }

    /// The cache bound under `launcher`, each operand a tile its kernel decodes as it reads it
    /// ([`Tile::decoded`](cubecl::tile::Tile::decoded)): the keys along the head dim `D`, the
    /// values along `DV`, and the keys' fix-up, the same words read through the sign codebook
    /// under the residual lengths, where they carry one.
    pub fn bind(&self, launcher: &Launcher) -> PackedCacheBinding {
        PackedCacheBinding {
            key: self.key.bind(launcher, self.bits.key, D),
            value: self.value.bind(launcher, self.bits.value, DV),
            fixup: self.fixup.as_ref().map(|fixup| {
                let fixup = PackedAttentionOperand {
                    words: self.key.words.clone(),
                    norms: fixup.errs.clone(),
                    codebook: fixup.sign_codebook.clone(),
                };
                fixup.bind(launcher, self.bits.key, D)
            }),
        }
    }

    /// What a launch over this cache compiles about it.
    pub fn layout(&self) -> PackedLayout {
        PackedLayout {
            bits: self.bits,
            fixup: match self.fixup {
                Some(_) => KeyFixup::Qjl,
                None => KeyFixup::RoundOnly,
            },
        }
    }
}

/// A packed cache bound for a launch ([`PackedAttentionCache::bind`]).
pub struct PackedCacheBinding {
    pub key: TileArgLaunch,
    pub value: TileArgLaunch,
    pub fixup: Option<TileArgLaunch>,
}

/// The axis of a codebook's table, its positions: outside every kernel's space.
const CODEBOOK: Axis = Axis(7);

/// What every packed binding rests on, checked by [`PackedAttentionCache::new`].
const CHECKED_BY_NEW: &str =
    "a packed cache's buffers are as PackedAttentionCache::new checks them";

impl PackedAttentionOperand {
    /// The operand bound under `launcher` along `[B, KV, S, head]`: its words' `bits`-wide fields
    /// indices into its codebook, each vector under its norm. Bound a word a line, the width a
    /// decoding read takes.
    fn bind(&self, launcher: &Launcher, bits: u32, head: Axis) -> TileArgLaunch {
        const TABLE: ElemType = ElemType::Float(FloatKind::F32);
        let axes = [B, KV, S, head];
        // A norm per vector, `[b, kv, s, 1]`, scales the vector along `[b, kv, s]`.
        let binding = &self.norms.binding;
        let rows = self.norms.viewed(Metadata::new(
            binding.shape[..3].to_vec(),
            binding.strides[..3].to_vec(),
        ));
        let norms = launcher
            .arg(rows.binding, TABLE)
            .axes(&axes[..3])
            .build()
            .expect(CHECKED_BY_NEW)
            .arg();
        let codebook = TileArgLaunch::new(
            self.codebook.binding.clone().into_tensor_arg(),
            TABLE,
            1,
            TileSpec::direct(&[CODEBOOK]),
        );
        // A packed binding is stated in the values its words hold.
        let per_word = 32 / bits as usize;
        let binding = &self.words.binding;
        let mut shape = binding.shape.to_vec();
        shape[3] *= per_word;
        let strides = binding.strides[..3]
            .iter()
            .map(|stride| stride * per_word)
            .chain([1])
            .collect::<Vec<_>>();
        let values = self.words.viewed(Metadata::new(shape, strides));
        launcher
            .arg(values.binding, ElemType::UInt(UIntKind::U32))
            .axes(&axes)
            .vectorize(1)
            .packed(Field::Index {
                bits: bits as usize,
            })
            .build()
            .expect(CHECKED_BY_NEW)
            .arg()
            .codebook(codebook)
            .expect(CHECKED_BY_NEW)
            .scales(norms)
    }
}
