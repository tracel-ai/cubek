//! A packed cache as the attention reads it: each operand's codebook indices packed into words,
//! its per-vector norms and the codebook, and the keys' QJL fix-up where there is one.

use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::PackedCacheArgLaunch;
use cubek_microkernel::attention::packed_cache::{KeyFixup, KvBitWidths};

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

impl PackedAttentionCache {
    /// A packed cache of `key` and `value` at `bits`, the keys carrying `fixup` where there is one.
    ///
    /// # Panics
    ///
    /// Words that are not `u32` or whose fields of a row are not adjacent, or a norm, codebook or
    /// fix-up table not in `f32`: what a caller converts before handing the cache over.
    pub fn new(
        key: PackedAttentionOperand,
        value: PackedAttentionOperand,
        bits: KvBitWidths,
        fixup: Option<QjlFixup>,
    ) -> Self {
        const WORD: ElemType = ElemType::UInt(UIntKind::U32);
        const TABLE: ElemType = ElemType::Float(FloatKind::F32);
        for operand in [&key, &value] {
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

    /// The cache as a kernel argument
    /// ([`PackedCacheArg`](cubek_microkernel::attention::PackedCacheArg)).
    pub fn arg(&self) -> PackedCacheArgLaunch {
        // Without a fix-up the kernel still takes its two tensors, and never reads them.
        let (errs, sign_codebook) = match &self.fixup {
            Some(fixup) => (&fixup.errs, &fixup.sign_codebook),
            None => (&self.key.norms, &self.key.codebook),
        };
        let arg = |tensor: &PlainTensor| tensor.binding.clone().into_tensor_arg();
        PackedCacheArgLaunch::new(
            arg(&self.key.words),
            arg(&self.key.norms),
            arg(&self.key.codebook),
            arg(errs),
            arg(sign_codebook),
            arg(&self.value.words),
            arg(&self.value.norms),
            arg(&self.value.codebook),
            self.coords(),
            self.bits.key,
            self.bits.value,
            self.layout().fixup,
        )
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
