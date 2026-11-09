//! Where an attention reads its keys and values from.

use crate::kernels::{AttentionCacheLayout, PackedAttentionCache};
use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::tensor::PlainTensor;

/// What an attention routine declines where the keys and values are read through an index list.
pub const INDEXED_KEYS_UNREAD: Refusal = Refusal::Declined {
    why: "the attention routines read every attended key, not an index list of them",
};

/// What an attention routine declines where a packed cache lies in pools of pages.
pub const PAGED_PACKED_UNREAD: Refusal = Refusal::Declined {
    why: "the packed attention reads a packed cache laid out as sequences, not in pages",
};

/// The keys and values an attention reads, and where each user's positions lie in them.
///
/// Three independent choices, each of which combines with the others:
/// - what is stored, the [`cache`](Self::cache): float keys and values, or a packed cache;
/// - where a position lies, the [`pages`](Self::pages): along each user's own positions, or in pools
///   of pages a table names;
/// - which positions a user reads, the [`indices`](Self::indices): every attended one, or the
///   ones its list names.
///
/// A source only says how a tile of positions `[p, p + T)` of user `b` and key head `h` is
/// loaded: the attention gives the same answer wherever the pages break, so neither the pages nor
/// the index list is an axis of the attention's space.
#[derive(Debug, Clone)]
pub struct KvSource {
    pub cache: AttentionCache,
    /// The `u32` table `[b, pages_per_sequence]` naming each user's pages in order: user `b`'s
    /// position `s` lies in page `pages[b, s / page_size]`, and the cache's buffers are pools of
    /// pages, `[pages, kv, page_size, ·]`. `None` where the buffers are `[b, kv, s, ·]`.
    pub pages: Option<PlainTensor>,
    /// The `u32` positions `[b, n]` each user reads, in order. `None` where a user reads every
    /// position its length attends.
    pub indices: Option<PlainTensor>,
}

/// The keys and values an attention stores.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, held by value as the launch's other operands are; both arms are tensors"
)]
pub enum AttentionCache {
    /// Float keys `[·, kv, s, d]` and values `[·, kv, s, dv]`.
    Float {
        key: PlainTensor,
        value: PlainTensor,
    },
    /// A packed cache, decoded as it is read.
    Packed(PackedAttentionCache),
}

impl KvSource {
    /// `cache`, laid out `[b, kv, s, ·]`, every attended position read.
    pub fn new(cache: AttentionCache) -> Self {
        Self {
            cache,
            pages: None,
            indices: None,
        }
    }

    /// The same source, its buffers pools of pages named by `pages` ([`pages`](Self::pages)).
    pub fn paged(self, pages: PlainTensor) -> Self {
        Self {
            pages: Some(pages),
            ..self
        }
    }

    /// The same source, each user reading the positions `indices` lists
    /// ([`indices`](Self::indices)).
    pub fn indexed(self, indices: PlainTensor) -> Self {
        Self {
            indices: Some(indices),
            ..self
        }
    }

    /// Key heads.
    pub(crate) fn num_kv_heads(&self) -> usize {
        match &self.cache {
            AttentionCache::Float { key, .. } => key.shape()[1],
            AttentionCache::Packed(cache) => cache.key.words.shape()[1],
        }
    }

    /// Keys each user holds, attended or not: a pool's are its table's pages.
    pub(crate) fn seq_kv(&self) -> usize {
        let positions = match &self.cache {
            AttentionCache::Float { key, .. } => key.shape()[2],
            AttentionCache::Packed(cache) => cache.key.words.shape()[2],
        };
        match &self.pages {
            Some(pages) => pages.shape()[1] * positions,
            None => positions,
        }
    }

    /// Elements of a value.
    pub(crate) fn val_dim(&self) -> usize {
        match &self.cache {
            AttentionCache::Float { value, .. } => value.shape()[3],
            AttentionCache::Packed(cache) => cache.coords(),
        }
    }

    /// How the keys and values lie, for a source the routines read: no index list, and a packed
    /// cache laid out as sequences.
    pub(crate) fn layout(&self) -> AttentionCacheLayout {
        match (&self.cache, &self.pages) {
            (AttentionCache::Packed(cache), _) => AttentionCacheLayout::Packed(cache.layout()),
            (AttentionCache::Float { .. }, None) => AttentionCacheLayout::Sequences,
            (AttentionCache::Float { key, .. }, Some(_)) => AttentionCacheLayout::Pages {
                page_size: key.shape()[2],
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::*;
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::zspace::Shape;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_test_utils::TestInput;

    /// An index list and a packed cache in pages are each declined by the one entry, before
    /// anything is launched; the same queries and keys read contiguously are admitted.
    #[test]
    fn the_entry_declines_what_no_routine_reads() {
        let client: Client = cubecl::test_device().client();
        let tensor = |shape: &[usize], element: ElemType| {
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                    .dtype(element)
                    .zeros()
                    .generate_without_host_data(),
            )
        };
        let (f32, u32) = (f32::elem_type_native(), u32::elem_type_native());
        let lengths = || tensor(&[1], u32);
        let dense = || AttentionQueries::Dense {
            query: tensor(&[1, 4, 3, 8], f32),
            lengths: lengths(),
        };
        let float = || {
            KvSource::new(AttentionCache::Float {
                key: tensor(&[1, 2, 16, 8], f32),
                value: tensor(&[1, 2, 16, 8], f32),
            })
        };
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, 8);
        let attention = |queries, kv| {
            AttentionTensors::attention(&client, queries, kv, scoring)
                .map(|_| ())
                .err()
        };

        assert_eq!(attention(dense(), float()), None);
        assert_eq!(
            attention(dense(), float().indexed(tensor(&[1, 4], u32))),
            Some(INDEXED_KEYS_UNREAD)
        );
        let operand = || PackedAttentionOperand {
            words: tensor(&[2, 2, 4, 1], u32),
            norms: tensor(&[2, 2, 4, 1], f32),
            codebook: tensor(&[16], f32),
        };
        let bits = KvBitWidths { key: 4, value: 4 };
        let packed = PackedAttentionCache::new(operand(), operand(), bits, None);
        let paged_packed =
            KvSource::new(AttentionCache::Packed(packed)).paged(tensor(&[1, 2], u32));
        assert_eq!(attention(dense(), paged_packed), Some(PAGED_PACKED_UNREAD));
    }
}
