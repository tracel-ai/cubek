//! Where an attention reads its keys and values from.

use super::kernel::block_pages_kernel;
use crate::kernels::{AttentionCacheLayout, AttentionVisibility, PackedAttentionCache};
use core::num::NonZeroUsize;
use cubecl::calculate_cube_count_elemwise;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;

/// What an attention routine declines where a packed cache is read through an index list.
pub const INDEXED_KEYS_UNREAD: Refusal = Refusal::Declined {
    why: "the packed attention reads every attended key of a packed cache, not an index list",
};

/// What an attention routine declines where the keys read through an index list are hidden from
/// a query by their position: a listed key's position is its slot in the list, not the cache's.
pub const INDEXED_VISIBILITY_UNREAD: Refusal = Refusal::Declined {
    why: "keys read through an index list are attended whole: no mask places them by position",
};

/// What an attention routine declines where a listed block of keys would straddle two pages, or
/// the key and value buffers would place the blocks apart.
pub const KEY_BLOCK_SPLITS_A_PAGE: Refusal = Refusal::Declined {
    why: "a block of listed keys lies in one page, and at the same place of the key and value \
          buffers",
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
/// - which positions a user reads, the [`blocks`](Self::blocks): every attended one, or the
///   blocks of consecutive positions its list names.
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
    /// The blocks each user reads, in order. `None` where a user reads every position its length
    /// attends.
    pub blocks: Option<KeyBlocks>,
}

/// The blocks of consecutive positions each user reads, in order: its keys are the blocks' keys
/// laid end to end, and its length counts them.
#[derive(Debug, Clone)]
pub struct KeyBlocks {
    /// The `u32` blocks `[b, n]`: user `b`'s `j`-th block is the positions
    /// `table[b, j]·size .. (table[b, j] + 1)·size` of its cache.
    pub table: PlainTensor,
    /// Positions a block holds.
    pub size: NonZeroUsize,
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
            blocks: None,
        }
    }

    /// The same source, its buffers pools of pages named by `pages` ([`pages`](Self::pages)).
    pub fn paged(self, pages: PlainTensor) -> Self {
        Self {
            pages: Some(pages),
            ..self
        }
    }

    /// The same source, each user reading the blocks of `size` positions `table` lists
    /// ([`blocks`](Self::blocks)).
    pub fn indexed(self, table: PlainTensor, size: NonZeroUsize) -> Self {
        Self {
            blocks: Some(KeyBlocks { table, size }),
            ..self
        }
    }

    /// This source read as pools of pages of one block each, where it lists blocks, `visibility`
    /// the one the attention reading it scores at: each buffer viewed as blocks of its own
    /// positions, and a table naming each user's listed blocks in order, composed with the cache's
    /// page table where the cache lies in pages. A user's cache laid out as a sequence is its one
    /// page. The key head of a block is a stride of the view, so every query of a group reads the
    /// keys of its head, and a block's positions are read as they lie, contiguous within it. A
    /// source that lists no blocks is as it is.
    ///
    /// # Errors
    ///
    /// [`INDEXED_KEYS_UNREAD`] over a packed cache, [`INDEXED_VISIBILITY_UNREAD`] at any
    /// visibility but [`AttentionVisibility::Attended`], and [`KEY_BLOCK_SPLITS_A_PAGE`] where a
    /// block straddles a page or the key and value buffers place their blocks apart.
    pub fn read_as_block_pages(
        self,
        client: &Client,
        visibility: AttentionVisibility,
    ) -> Result<Self, Refusal> {
        let Some(blocks) = self.blocks else {
            return Ok(self);
        };
        let AttentionCache::Float { key, value } = self.cache else {
            return Err(INDEXED_KEYS_UNREAD);
        };
        if visibility != AttentionVisibility::Attended {
            return Err(INDEXED_VISIBILITY_UNREAD);
        }
        let pages = self.pages.unwrap_or_else(|| {
            let users = key.shape()[0];
            let pages: Vec<u32> = (0..users as u32).collect();
            PlainTensor::of_u32s(client, &[users, 1], &pages)
        });
        let key = BlockPool::of(&key, blocks.size)?;
        let value = BlockPool::of(&value, blocks.size)?;
        if key.page_blocks != value.page_blocks || key.blocks_per_page != value.blocks_per_page {
            return Err(KEY_BLOCK_SPLITS_A_PAGE);
        }
        Ok(Self {
            pages: Some(key.table(client, &blocks.table, &pages)),
            cache: AttentionCache::Float {
                key: key.view,
                value: value.view,
            },
            blocks: None,
        })
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

    /// How the keys and values lie, for a source the routines read: no blocks listed, and a packed
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

/// One buffer of a cache viewed as a pool of pages of one block each: `[blocks, kv, size, ·]`,
/// its first dim stepping a block of positions of the cache's first key head, its key heads a
/// stride on, so a block and a key head name the block's positions of that head.
struct BlockPool {
    view: PlainTensor,
    /// Blocks a page of the cache holds.
    blocks_per_page: usize,
    /// Blocks the view steps from the start of one page of the cache to the next.
    page_blocks: usize,
}

impl BlockPool {
    /// `buffer`, `[pages, kv, page_size, ·]`, viewed as blocks of `size` positions: as many as the
    /// listed blocks can reach, every block of every page's first key head.
    ///
    /// # Errors
    ///
    /// [`KEY_BLOCK_SPLITS_A_PAGE`] where `size` does not tile a page, or a page's stride is not a
    /// whole number of blocks.
    fn of(buffer: &PlainTensor, size: NonZeroUsize) -> Result<Self, Refusal> {
        let (shape, strides) = (buffer.shape(), &buffer.binding.strides);
        let size = size.get();
        let block_stride = size * strides[2];
        if !shape[2].is_multiple_of(size) || !strides[0].is_multiple_of(block_stride) {
            return Err(KEY_BLOCK_SPLITS_A_PAGE);
        }
        let blocks_per_page = shape[2] / size;
        let page_blocks = strides[0] / block_stride;
        let blocks = (shape[0] - 1) * page_blocks + blocks_per_page;
        let view = buffer.viewed(Metadata::new(
            Shape::from(vec![blocks, shape[1], size, shape[3]]),
            vec![block_stride, strides[1], strides[2], strides[3]],
        ));
        Ok(Self {
            view,
            blocks_per_page,
            page_blocks,
        })
    }

    /// The `u32` table `[b, n]` naming in this view each user's listed blocks, `blocks`
    /// (`[b, n]`), where `pages` (`[b, pages_per_sequence]`) names each user's pages of the
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
            self.blocks_per_page as u32,
            self.page_blocks as u32,
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
    use cubecl::std::tensor::TensorHandle;
    use cubecl::zspace::Shape;
    use cubecl::zspace::metadata::Metadata;
    use cubecl::zspace::striding::row_major_contiguous_strides;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_std::Hardware;
    use cubek_test_utils::{HostData, HostDataType, TestInput, ValidationResult};

    /// Packed queries, a packed cache in pages, and blocks of keys listed over a packed cache, at a
    /// causal visibility or straddling a page are each declined by the one entry, before any
    /// routine is launched; the same queries and keys stated densely, or as listed blocks, are
    /// admitted.
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
        let packed_queries = AttentionQueries::Packed {
            query: tensor(&[3, 4, 8], f32),
            ends: tensor(&[1], u32),
            longest: 3,
            lengths: lengths(),
        };
        assert_eq!(
            attention(packed_queries, float()),
            Some(PACKED_QUERIES_UNREAD)
        );
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
        let paged_packed =
            KvSource::new(AttentionCache::Packed(packed.clone())).paged(tensor(&[1, 2], u32));
        assert_eq!(attention(dense(), paged_packed), Some(PAGED_PACKED_UNREAD));
        let listed_packed =
            KvSource::new(AttentionCache::Packed(packed)).indexed(tensor(&[1, 2], u32), block(4));
        assert_eq!(attention(dense(), listed_packed), Some(INDEXED_KEYS_UNREAD));
    }

    /// `values`, laid out row major as `shape`, on `client`'s device.
    fn floats(client: &Client, shape: &[usize], values: &[f32]) -> PlainTensor {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let meta = Metadata::new(
            Shape::from(shape.to_vec()),
            row_major_contiguous_strides(shape),
        );
        PlainTensor::of_handle(TensorHandle::from_metadata(
            client.create_from_slice(&bytes),
            meta,
            f32::elem_type_native(),
        ))
    }

    /// Attention over listed blocks of keys reads what attention over the same keys gathered into
    /// a cache of their own reads, from a cache laid out as sequences and from the same cache in
    /// pools of pages, at the lead of every kernel that serves the problem: two users of two key
    /// heads a group of two, each listing three blocks of four of its 32 positions, out of order,
    /// and attending 12 and 9 of them.
    #[test]
    fn listed_blocks_read_as_their_keys_gathered() {
        const USERS: usize = 2;
        const KV: usize = 2;
        const GROUP: usize = 2;
        const DIM: usize = 16;
        const SEQ: usize = 32;
        const PAGE: usize = 8;
        const BLOCK: usize = 4;
        const LISTED: [[u32; 3]; USERS] = [[5, 0, 6], [2, 7, 3]];
        // Each user's pages of the pool, in order.
        const PAGES: [[u32; SEQ / PAGE]; USERS] = [[3, 0, 6, 1], [7, 2, 5, 4]];
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
            random(vec![USERS, KV, SEQ, DIM], 1),
            random(vec![USERS, KV, SEQ, DIM], 2),
        );
        let query = random(vec![USERS, KV * GROUP, 1, DIM], 3);
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
            floats(&client, shape, &values)
        };
        let listed_slots = LISTED[0].len() * BLOCK;
        let gathered = |host: &HostData| {
            laid(&[USERS, KV, listed_slots, DIM], &|at| {
                let position = LISTED[at[0]][at[2] / BLOCK] as usize * BLOCK + at[2] % BLOCK;
                host.get_f32(&[at[0], at[1], position, at[3]])
            })
        };
        let sequences = |host: &HostData| laid(&[USERS, KV, SEQ, DIM], &|at| host.get_f32(at));
        let pool = |host: &HostData| {
            laid(&[USERS * SEQ / PAGE, KV, PAGE, DIM], &|at| {
                let (user, column) = (0..USERS)
                    .flat_map(|user| (0..SEQ / PAGE).map(move |column| (user, column)))
                    .find(|&(user, column)| PAGES[user][column] as usize == at[0])
                    .expect("every page of the pool is a user's");
                host.get_f32(&[user, at[1], column * PAGE + at[2], at[3]])
            })
        };
        let u32s = |shape: &[usize], values: &[u32]| PlainTensor::of_u32s(&client, shape, values);
        let lengths = || u32s(&[USERS], &[12, 9]);
        let listed = || u32s(&[USERS, LISTED[0].len()], LISTED.as_flattened());
        let block = NonZeroUsize::new(BLOCK).unwrap();
        let queries = || AttentionQueries::Dense {
            query: laid(&[USERS, KV * GROUP, 1, DIM], &|at| query.get_f32(at)),
            lengths: lengths(),
        };
        let float = |key, value| KvSource::new(AttentionCache::Float { key, value });
        let scoring = AttentionScoring::standard(AttentionVisibility::Attended, DIM);
        let mut served = 0;
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
                float(sequences(&key), sequences(&value)).indexed(listed(), block),
                float(pool(&key), pool(&value))
                    .paged(u32s(&[USERS, SEQ / PAGE], PAGES.as_flattened()))
                    .indexed(listed(), block),
            ];
            for source in listed_sources {
                let actual = run(source).unwrap_or_else(|refusal| {
                    panic!("{kernel:?} reads the gathered keys, and declines the listed: {refusal}")
                });
                let compared = cubek_test_utils::assert_equals_approx(&actual, &expected, 1e-4);
                assert!(
                    matches!(compared, ValidationResult::Pass),
                    "{kernel:?}: {compared:?}"
                );
            }
            served += 1;
        }
        assert!(served > 0, "the unit routine serves any problem");
    }
}
