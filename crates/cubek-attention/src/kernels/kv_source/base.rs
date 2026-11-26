//! Where an attention reads its keys and values from.

use super::kernel::block_pages_kernel;
use crate::kernels::{AttentionCacheLayout, AttentionVisibility, PackedAttentionCache};
#[cfg(doc)]
use crate::kernels::{PACKED_CACHE_UNREAD, PACKED_WIDTH_UNREAD};
use core::num::NonZeroUsize;
use cubecl::calculate_cube_count_elemwise;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Decline;
use cubecl::tile::{TileSpec, TileTensorLaunch};
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{B, S};

/// What an attention routine declines where a packed cache is read through an index list.
pub const INDEXED_KEYS_UNREAD: Decline = Decline::Unserved {
    why: "the packed attention reads every attended key of a packed cache, not an index list",
};

/// What an attention routine declines where the keys read through an index list are hidden from
/// a query by their position: a listed key's position is its slot in the list, not the cache's.
pub const INDEXED_VISIBILITY_UNREAD: Decline = Decline::Unserved {
    why: "keys read through an index list are attended whole: no mask places them by position",
};

/// What an attention routine declines where a listed block of keys would straddle two pages, or
/// the key and value buffers would place the blocks apart.
pub const KEY_BLOCK_SPLITS_A_PAGE: Decline = Decline::Unserved {
    why: "a block of listed keys lies in one page, and at the same place of the key and value \
          buffers",
};

/// What an attention routine declines where listed blocks of keys have no place of their own: a
/// buffer of no page or sequence, or one whose positions lie at stride zero.
pub const KEY_BLOCKS_UNPLACED: Decline = Decline::Unserved {
    why: "listed blocks of keys lie in a buffer of at least one page or sequence, its positions \
          apart",
};

/// What an attention routine declines where a packed cache is stated under per-tensor scales: its
/// keys and values are read under the norms it stores.
pub const SCALED_PACKED_UNREAD: Decline = Decline::Unserved {
    why: "a packed cache is read under the norms it stores, not under per-tensor scales",
};

/// What an attention declines before preparing anything: float keys and values that are no
/// `[b, kv, s, ·]`, or `[pages, kv, page_size, ·]`, agreeing on all but their last dim.
pub const CACHE_SHAPES_DISAGREE: Decline = Decline::Unserved {
    why: "keys and values are of rank four, and agree on their sequences or pages, key heads and \
          positions",
};

/// What an attention declines before preparing anything: a table of pages or of listed blocks that
/// is no `u32` row a sequence, none empty.
pub const PAGE_TABLE_UNREAD: Decline = Decline::Unserved {
    why: "a page or block table is u32 of rank two, one non-empty row a sequence, over a pool of \
          non-empty pages",
};

/// What an attention declines before preparing anything: a cache of other sequences than the
/// query's.
pub const CACHE_SEQUENCES_DISAGREE: Decline = Decline::Unserved {
    why: "a cache laid out as sequences holds one row a sequence of the query",
};

/// What an attention declines before preparing anything: a cache scale that is not one `f32` over
/// its whole side.
pub const CACHE_SCALE_UNREAD: Decline = Decline::Unserved {
    why: "a cache's scale is one f32 over its whole side",
};

/// The keys and values an attention reads, and where each sequence's positions lie in them.
///
/// Four choices, of which the entry declines the combinations no routine reads
/// ([`AttentionTensors::attention`](crate::kernels::AttentionTensors::attention)):
/// - what is stored, the [`cache`](Self::cache): float keys and values, or a packed cache;
/// - what each side of a float cache is multiplied by, the [`scales`](Self::scales): nothing, or
///   one scale over all its keys or all its values;
/// - where a position lies, the [`pages`](Self::pages): along each sequence's own positions, or in
///   pools of pages a table names;
/// - which positions a sequence reads, the [`blocks`](Self::blocks): every attended one, or the
///   blocks of consecutive positions its list names.
///
/// A source only says how a tile of positions `[p, p + T)` of sequence `b` and key head `h` is
/// loaded: the attention gives the same answer wherever the pages break, so neither the pages nor
/// the index list is a dimension of the attention's space.
#[derive(Debug, Clone)]
pub struct KvSource {
    pub cache: AttentionCache,
    /// The `u32` table `[b, pages_per_sequence]` naming each sequence's pages in order: sequence
    /// `b`'s position `s` lies in page `pages[b, s / page_size]`, and the cache's buffers are pools
    /// of pages, `[pages, kv, page_size, ·]`, a packed cache's words, norms and residual lengths
    /// alike. `None` where the buffers are `[b, kv, s, ·]`.
    pub pages: Option<PlainTensor>,
    /// The blocks each sequence reads, in order. `None` where a sequence reads every position its
    /// length attends.
    pub blocks: Option<KeyBlocks>,
    /// The scale each side of a float cache is stored under.
    pub scales: CacheScales,
}

/// The one scale each side of a float cache is stored under, over the whole tensor: the keys the
/// attention reads are the stored keys times [`key`](Self::key), its values the stored values
/// times [`value`](Self::value), as an FP8 cache quantized per tensor stores them. The routines
/// read the stored keys and values as they lie and apply each scale once, to the sums it enters.
#[derive(Debug, Clone, Default)]
pub struct CacheScales {
    /// The one-element `f32` tensor every key is multiplied by. `None` where the keys are read as
    /// stored.
    pub key: Option<PlainTensor>,
    /// The one-element `f32` tensor every value is multiplied by. `None` where the values are read
    /// as stored.
    pub value: Option<PlainTensor>,
}

impl CacheScales {
    /// Whether either side is stored under a scale.
    pub fn any(&self) -> bool {
        self.key.is_some() || self.value.is_some()
    }
}

/// The blocks of consecutive positions each sequence reads, in order: its keys are the blocks' keys
/// laid end to end, and its length counts them.
#[derive(Debug, Clone)]
pub struct KeyBlocks {
    /// The `u32` blocks `[b, n]`: sequence `b`'s `j`-th block is the positions
    /// `table[b, j]·size .. (table[b, j] + 1)·size` of its cache.
    pub table: PlainTensor,
    /// Positions a block holds.
    pub size: NonZeroUsize,
}

/// The keys and values an attention stores.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, kept by value as the launch's other operands are; both arms are tensors"
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
            blocks: None,
            scales: CacheScales::default(),
        }
    }

    /// The same source, its buffers pools of pages named by `pages` ([`pages`](Self::pages)).
    pub fn paged(self, pages: PlainTensor) -> Self {
        Self {
            pages: Some(pages),
            ..self
        }
    }

    /// The same source, each sequence reading the blocks of `size` positions `table` lists
    /// ([`blocks`](Self::blocks)).
    pub fn indexed(self, table: PlainTensor, size: NonZeroUsize) -> Self {
        Self {
            blocks: Some(KeyBlocks { table, size }),
            ..self
        }
    }

    /// The same source, each side of its float cache stored under its scale of `scales`
    /// ([`scales`](Self::scales)).
    pub fn scaled(self, scales: CacheScales) -> Self {
        Self { scales, ..self }
    }

    /// This source read as pools of pages of one block each, where it lists blocks, `visibility`
    /// the one the attention reading it scores at: each buffer viewed as blocks of its own
    /// positions, and a table naming each sequence's listed blocks in order, composed with the
    /// cache's page table where the cache lies in pages. A cache laid out as sequences is each
    /// sequence's one page. The key head of a block is a stride of the view, so every query of a
    /// group reads the keys of its head, and a block's positions are read as they lie, contiguous
    /// within it. A source that lists no blocks is as it is.
    ///
    /// # Guarantees
    ///
    /// Whatever the tables hold, the composed table names no block past the view: a listed block
    /// past its sequence's pages lies in their last page, and a page past the pool is its last
    /// page. A decline comes before anything is allocated or launched.
    ///
    /// # Errors
    ///
    /// [`INDEXED_KEYS_UNREAD`] over a packed cache, [`INDEXED_VISIBILITY_UNREAD`] at any visibility
    /// but [`AttentionVisibility::Attended`], [`KEY_BLOCK_SPLITS_A_PAGE`] where a block
    /// straddles a page or the key and value buffers place their blocks apart,
    /// [`KEY_BLOCKS_UNPLACED`] where a buffer leaves a block no place of its own, and
    /// [`CACHE_SHAPES_DISAGREE`], [`PACKED_CACHE_UNREAD`], [`PACKED_WIDTH_UNREAD`],
    /// [`PAGE_TABLE_UNREAD`] and [`CACHE_SCALE_UNREAD`] for buffers no source holds.
    pub fn read_as_block_pages(
        self,
        client: &Client,
        visibility: AttentionVisibility,
    ) -> Result<Self, Decline> {
        let pools = self.block_pools(visibility)?;
        Ok(self.into_block_pages(client, pools))
    }
}

impl KvSource {
    /// The pools of one block each this source's buffers are viewed as, where it lists blocks:
    /// every check [`read_as_block_pages`](Self::read_as_block_pages) makes, nothing allocated.
    ///
    /// # Errors
    ///
    /// As [`read_as_block_pages`](Self::read_as_block_pages).
    pub(crate) fn block_pools(
        &self,
        visibility: AttentionVisibility,
    ) -> Result<Option<BlockPools>, Decline> {
        self.admit()?;
        let Some(blocks) = &self.blocks else {
            return Ok(None);
        };
        let AttentionCache::Float { key, value } = &self.cache else {
            return Err(INDEXED_KEYS_UNREAD);
        };
        if visibility != AttentionVisibility::Attended {
            return Err(INDEXED_VISIBILITY_UNREAD);
        }
        let sequences = self
            .pages
            .as_ref()
            .map_or(key.shape()[0], |pages| pages.shape()[0]);
        admit_table(&blocks.table, sequences)?;
        let key_pool = BlockPool::of(key, blocks.size)?;
        let value_pool = BlockPool::of(value, blocks.size)?;
        let placed = |pool: &BlockPool| (pool.page_step_block_count, pool.page_block_count);
        if placed(&key_pool) != placed(&value_pool) {
            return Err(KEY_BLOCK_SPLITS_A_PAGE);
        }
        Ok(Some(BlockPools {
            key: key_pool,
            value: value_pool,
            table: blocks.table.clone(),
        }))
    }

    /// This source read through `pools`, its [`block_pools`](Self::block_pools): as it is where
    /// there are none.
    pub(crate) fn into_block_pages(self, client: &Client, pools: Option<BlockPools>) -> Self {
        let Some(pools) = pools else {
            return self;
        };
        let pages = self.pages.unwrap_or_else(|| {
            let sequences = pools.key.page_count;
            let pages: Vec<u32> = (0..sequences as u32).collect();
            PlainTensor::of_u32s(client, &[sequences, 1], &pages)
        });
        Self {
            pages: Some(pools.key.table(client, &pools.table, &pages)),
            cache: AttentionCache::Float {
                key: pools.key.view,
                value: pools.value.view,
            },
            blocks: None,
            scales: self.scales,
        }
    }

    /// Whether the source's buffers are as it says they lie, whatever the query.
    ///
    /// # Errors
    ///
    /// As [`read_as_block_pages`](Self::read_as_block_pages), but for the listed blocks.
    fn admit(&self) -> Result<(), Decline> {
        match &self.cache {
            AttentionCache::Float { key, value } => Self::admit_float(key, value)?,
            AttentionCache::Packed(cache) => cache.admit()?,
        }
        if let Some(pages) = &self.pages {
            self.admit_pages(pages)?;
        }
        for scale in [&self.scales.key, &self.scales.value].into_iter().flatten() {
            if scale.elem != ElemType::Float(FloatKind::F32) || scale.shape().num_elements() != 1 {
                return Err(CACHE_SCALE_UNREAD);
            }
        }
        Ok(())
    }

    /// Whether `key` and `value` agree on all but their last dim.
    ///
    /// # Errors
    ///
    /// [`CACHE_SHAPES_DISAGREE`].
    fn admit_float(key: &PlainTensor, value: &PlainTensor) -> Result<(), Decline> {
        let (key, value) = (key.shape(), value.shape());
        if key.rank() != 4 || value.rank() != 4 || key[..3] != value[..3] {
            return Err(CACHE_SHAPES_DISAGREE);
        }
        Ok(())
    }

    /// Whether `pages` is a table of rank two over the cache's pools: the pool, its pages and the
    /// table's rows not empty.
    ///
    /// # Errors
    ///
    /// [`PAGE_TABLE_UNREAD`].
    fn admit_pages(&self, pages: &PlainTensor) -> Result<(), Decline> {
        let [pool, page_size] = self.rows();
        if pool == 0 || page_size == 0 || pages.shape().rank() != 2 {
            return Err(PAGE_TABLE_UNREAD);
        }
        admit_table(pages, pages.shape()[0])
    }

    /// Whether the source holds one row a sequence of `sequences`: of its buffers, or of the table
    /// that names its pages.
    ///
    /// # Errors
    ///
    /// [`PAGE_TABLE_UNREAD`] and [`CACHE_SEQUENCES_DISAGREE`].
    pub(crate) fn admit_sequences(&self, sequences: usize) -> Result<(), Decline> {
        match &self.pages {
            Some(pages) => admit_table(pages, sequences),
            None if self.rows()[0] == sequences => Ok(()),
            None => Err(CACHE_SEQUENCES_DISAGREE),
        }
    }

    /// What the cache's buffers hold, `[sequences or pages, positions]`.
    fn rows(&self) -> [usize; 2] {
        let buffer = match &self.cache {
            AttentionCache::Float { key, .. } => key,
            AttentionCache::Packed(cache) => &cache.key.words,
        };
        let shape = buffer.shape();
        [shape[0], shape[2]]
    }

    /// Key heads.
    pub(crate) fn num_kv_heads(&self) -> usize {
        match &self.cache {
            AttentionCache::Float { key, .. } => key.shape()[1],
            AttentionCache::Packed(cache) => cache.key.words.shape()[1],
        }
    }

    /// Keys each sequence holds, attended or not: a pool's are its table's pages.
    pub(crate) fn seq_kv(&self) -> usize {
        let [_, positions] = self.rows();
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

    /// How the keys and values lie, for a source the routines read, listing no blocks. A packed
    /// cache is packed wherever it lies: its pages are its binding's, which reads them through the
    /// table as it decodes ([`PackedAttentionCache::bind`]).
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

/// Whether `table` is `u32` of rank two, a non-empty row for each of `sequences`.
///
/// # Errors
///
/// [`PAGE_TABLE_UNREAD`].
fn admit_table(table: &PlainTensor, sequences: usize) -> Result<(), Decline> {
    let shape = table.shape();
    let rows = shape.rank() == 2 && shape[0] == sequences && shape[1] > 0;
    if !rows || table.elem != ElemType::UInt(UIntKind::U32) {
        return Err(PAGE_TABLE_UNREAD);
    }
    Ok(())
}

/// The page table `pages` ([`KvSource::pages`]) bound for a launch: keyed by the batch and the
/// keys' positions, it covers the launch, its rows the batch and its columns the keys.
pub(crate) fn page_table(pages: &PlainTensor) -> TileTensorLaunch {
    TileTensorLaunch::new(
        pages.binding.clone().into_tensor_arg(),
        ElemType::UInt(UIntKind::U32),
        1,
        TileSpec::direct(&[B, S]),
    )
}

/// A source's key and value buffers each viewed as a pool of pages of one block each
/// ([`KvSource::block_pools`]), with the listed blocks they read.
pub(crate) struct BlockPools {
    pub key: BlockPool,
    pub value: BlockPool,
    /// The `u32` blocks `[b, n]` each sequence reads ([`KeyBlocks::table`]).
    pub table: PlainTensor,
}

/// One buffer of a cache viewed as a pool of pages of one block each: `[blocks, kv, size, ·]`,
/// its first dim stepping a block of positions of the cache's first key head, its key heads a
/// stride on, so a block and a key head name the block's positions of that head.
pub(crate) struct BlockPool {
    pub view: PlainTensor,
    /// Blocks a page of the cache holds.
    pub page_block_count: usize,
    /// Blocks the view steps from the start of one page of the cache to the next.
    pub page_step_block_count: usize,
    /// Pages, or sequences, the buffer holds.
    pub page_count: usize,
}

impl BlockPool {
    /// `buffer`, `[pages, kv, page_size, ·]`, viewed as blocks of `size` positions: as many as the
    /// listed blocks can reach, every block of every page's first key head.
    ///
    /// # Errors
    ///
    /// [`KEY_BLOCKS_UNPLACED`] where there is no page or the positions lie at stride zero, and
    /// [`KEY_BLOCK_SPLITS_A_PAGE`] where `size` does not tile a page or a page's stride is not a
    /// whole number of blocks.
    fn of(buffer: &PlainTensor, size: NonZeroUsize) -> Result<Self, Decline> {
        let (shape, strides) = (buffer.shape(), &buffer.binding.strides);
        let size = size.get();
        let block_stride = size * strides[2];
        if shape[0] == 0 || block_stride == 0 {
            return Err(KEY_BLOCKS_UNPLACED);
        }
        if !shape[2].is_multiple_of(size) || !strides[0].is_multiple_of(block_stride) {
            return Err(KEY_BLOCK_SPLITS_A_PAGE);
        }
        let page_block_count = shape[2] / size;
        let page_step_block_count = strides[0] / block_stride;
        let blocks = (shape[0] - 1) * page_step_block_count + page_block_count;
        let view = buffer.viewed(Metadata::new(
            Shape::from(vec![blocks, shape[1], size, shape[3]]),
            vec![block_stride, strides[1], strides[2], strides[3]],
        ));
        Ok(Self {
            view,
            page_block_count,
            page_step_block_count,
            page_count: shape[0],
        })
    }

    /// The `u32` table `[b, n]` naming in this view each sequence's listed blocks, `blocks`
    /// (`[b, n]`), where `pages` (`[b, pages_per_sequence]`) names each sequence's pages of the
    /// cache.
    fn table(&self, client: &Client, blocks: &PlainTensor, pages: &PlainTensor) -> PlainTensor {
        let shape = blocks.shape();
        let (_, table) =
            PlainTensor::empty_contiguous(client, shape.clone(), ElemType::UInt(UIntKind::U32));
        let units = shape[0] * shape[1];
        let cube_dim = CubeDim::new(client, units);
        block_pages_kernel::launch(
            client,
            calculate_cube_count_elemwise(client, units, cube_dim),
            cube_dim,
            blocks.binding.clone().into_tensor_arg(),
            pages.binding.clone().into_tensor_arg(),
            table.binding.clone().into_tensor_arg(),
            shape[1] as u32,
            self.page_block_count as u32,
            self.page_step_block_count as u32,
            self.page_count as u32 - 1,
        );
        table
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::*;
    use core::num::NonZeroUsize;
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::zspace::Shape;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_std::Hardware;
    use cubek_test_utils::{HostData, HostDataType, TestInput, ValidationResult};

    /// A packed cache under per-tensor scales, and blocks of keys listed over a packed cache, at a
    /// causal visibility or straddling a page, are each declined by the one entry, before any
    /// routine is launched; the same queries and keys read contiguously, as listed blocks, or
    /// under per-tensor scales, and a packed cache in pages, are admitted.
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
        let block = |size| NonZeroUsize::new(size).unwrap();
        assert_eq!(
            attention(dense(), float().indexed(tensor(&[1, 2], u32), block(4))),
            None
        );
        assert_eq!(
            attention(dense(), float().indexed(tensor(&[1, 2], u32), block(3))),
            Some(KEY_BLOCK_SPLITS_A_PAGE)
        );
        let causal = AttentionScoring::standard(
            AttentionVisibility::Causal {
                queries: QueryPlacement::Last,
            },
            8,
        );
        assert_eq!(
            AttentionTensors::attention(
                &client,
                dense(),
                float().indexed(tensor(&[1, 2], u32), block(4)),
                causal
            )
            .err(),
            Some(INDEXED_VISIBILITY_UNREAD)
        );
        let operand = || PackedAttentionOperand {
            words: tensor(&[2, 2, 4, 1], u32),
            norms: tensor(&[2, 2, 4, 1], f32),
            codebook: tensor(&[16], f32),
        };
        let bits = KvBitWidths { key: 4, value: 4 };
        let packed = PackedAttentionCache::new(operand(), operand(), bits, None);
        let scales = || CacheScales {
            key: Some(tensor(&[1], f32)),
            value: Some(tensor(&[1], f32)),
        };
        assert_eq!(attention(dense(), float().scaled(scales())), None);
        let scaled_packed = KvSource::new(AttentionCache::Packed(packed.clone())).scaled(scales());
        assert_eq!(
            attention(dense(), scaled_packed),
            Some(SCALED_PACKED_UNREAD)
        );
        let paged_packed =
            KvSource::new(AttentionCache::Packed(packed.clone())).paged(tensor(&[1, 2], u32));
        assert_eq!(attention(dense(), paged_packed), None);
        let listed_packed =
            KvSource::new(AttentionCache::Packed(packed)).indexed(tensor(&[1, 2], u32), block(4));
        assert_eq!(attention(dense(), listed_packed), Some(INDEXED_KEYS_UNREAD));
    }

    /// Attention over listed blocks of keys reads what attention over the same keys gathered into a
    /// cache of their own reads, from a cache laid out as sequences and from the same cache in
    /// pools of pages, at the lead of every kernel that serves the problem: two sequences of two
    /// key heads a group of two, each listing three blocks of four of its 32 positions, out of
    /// order, and attending 12 and 9 of them.
    #[test]
    fn listed_blocks_read_as_their_keys_gathered() {
        const SEQUENCES: usize = 2;
        const KV: usize = 2;
        const GROUP: usize = 2;
        const DIM: usize = 16;
        const SEQ: usize = 32;
        const PAGE: usize = 8;
        const BLOCK: usize = 4;
        const LISTED: [[u32; 3]; SEQUENCES] = [[5, 0, 6], [2, 7, 3]];
        // Each sequence's pages of the pool, in order.
        const PAGES: [[u32; SEQ / PAGE]; SEQUENCES] = [[3, 0, 6, 1], [7, 2, 5, 4]];
        let client: Client = cubecl::test_device().client();
        let hw = Hardware::from_client(&client);
        let random = |shape: Vec<usize>, seed| {
            let (_, host) = TestInput::builder(client.clone(), Shape::from(shape))
                .dtype(f32::elem_type_native())
                .uniform(seed, -1.0, 1.0)
                .generate_with_f32_host_data();
            host
        };
        let (key, value) = (
            random(vec![SEQUENCES, KV, SEQ, DIM], 1),
            random(vec![SEQUENCES, KV, SEQ, DIM], 2),
        );
        let query = random(vec![SEQUENCES, KV * GROUP, 1, DIM], 3);
        let laid = |shape: &[usize], at: &dyn Fn(&[usize]) -> f32| {
            let mut values = Vec::new();
            for a in 0..shape[0] {
                for b in 0..shape[1] {
                    for c in 0..shape[2] {
                        for d in 0..shape[3] {
                            values.push(at(&[a, b, c, d]));
                        }
                    }
                }
            }
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                    .dtype(f32::elem_type_native())
                    .custom(values)
                    .generate_without_host_data(),
            )
        };
        let listed_slots = LISTED[0].len() * BLOCK;
        let gathered = |host: &HostData| {
            laid(&[SEQUENCES, KV, listed_slots, DIM], &|at| {
                let position = LISTED[at[0]][at[2] / BLOCK] as usize * BLOCK + at[2] % BLOCK;
                host.get_f32(&[at[0], at[1], position, at[3]])
            })
        };
        let contiguous = |host: &HostData| laid(&[SEQUENCES, KV, SEQ, DIM], &|at| host.get_f32(at));
        let pool = |host: &HostData| {
            laid(&[SEQUENCES * SEQ / PAGE, KV, PAGE, DIM], &|at| {
                let (sequence, column) = (0..SEQUENCES)
                    .flat_map(|sequence| (0..SEQ / PAGE).map(move |column| (sequence, column)))
                    .find(|&(sequence, column)| PAGES[sequence][column] as usize == at[0])
                    .expect("every page of the pool is a sequence's");
                host.get_f32(&[sequence, at[1], column * PAGE + at[2], at[3]])
            })
        };
        let u32s = |shape: &[usize], values: &[u32]| PlainTensor::of_u32s(&client, shape, values);
        let lengths = || u32s(&[SEQUENCES], &[12, 9]);
        let listed = || u32s(&[SEQUENCES, LISTED[0].len()], LISTED.as_flattened());
        let block = NonZeroUsize::new(BLOCK).unwrap();
        let queries = || AttentionQueries::Dense {
            query: laid(&[SEQUENCES, KV * GROUP, 1, DIM], &|at| query.get_f32(at)),
            lengths: lengths(),
        };
        let float = |key, value| KvSource::new(AttentionCache::Float { key, value });
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, DIM);
        let mut served = Vec::new();
        for kernel in AttentionKernel::all() {
            let run = |kv: KvSource| {
                AttentionTensors::attention(&client, queries(), kv, scoring)
                    .and_then(|tensors| kernel.launch_lead_on(&hw, tensors))
                    .map(|out| HostData::from_tensor_handle(&client, out, HostDataType::F32))
            };
            let Ok(expected) = run(float(gathered(&key), gathered(&value))) else {
                continue;
            };
            let listed_sources = [
                float(contiguous(&key), contiguous(&value)).indexed(listed(), block),
                float(pool(&key), pool(&value))
                    .paged(u32s(&[SEQUENCES, SEQ / PAGE], PAGES.as_flattened()))
                    .indexed(listed(), block),
            ];
            for source in listed_sources {
                let actual = run(source).unwrap_or_else(|decline| {
                    panic!("{kernel:?} reads the gathered keys, and declines the listed: {decline}")
                });
                let compared = cubek_test_utils::assert_equals_approx(&actual, &expected, 1e-4);
                assert!(
                    matches!(compared, ValidationResult::Pass),
                    "{kernel:?}: {compared:?}"
                );
            }
            served.push(kernel);
        }
        assert!(
            served.contains(&AttentionKernel::Unit),
            "the unit routine serves any problem: {served:?}"
        );
    }
}
