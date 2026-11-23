//! Where an attention reads its keys and values from.

use super::kernel::block_pages_kernel;
use crate::kernels::{AttentionCacheLayout, AttentionVisibility, PackedAttentionCache};
use core::num::NonZeroUsize;
use cubecl::calculate_cube_count_elemwise;
use cubecl::client::Client;
use cubecl::ir::{ElemType, UIntKind};
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

/// What an attention routine declines where a packed cache is stated under per-tensor scales: its
/// keys and values are read under the norms it stores.
pub const SCALED_PACKED_UNREAD: Decline = Decline::Unserved {
    why: "a packed cache is read under the norms it stores, not under per-tensor scales",
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
    /// # Errors
    ///
    /// [`INDEXED_KEYS_UNREAD`] over a packed cache, [`INDEXED_VISIBILITY_UNREAD`] at any visibility
    /// but [`AttentionVisibility::Attended`], and [`KEY_BLOCK_SPLITS_A_PAGE`] where a block
    /// straddles a page or the key and value buffers place their blocks apart.
    pub fn read_as_block_pages(
        self,
        client: &Client,
        visibility: AttentionVisibility,
    ) -> Result<Self, Decline> {
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
            let sequences = key.shape()[0];
            let pages: Vec<u32> = (0..sequences as u32).collect();
            PlainTensor::of_u32s(client, &[sequences, 1], &pages)
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
            scales: self.scales,
        })
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
    fn of(buffer: &PlainTensor, size: NonZeroUsize) -> Result<Self, Decline> {
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

    /// Sequences of the packed cases.
    const PACKED_SEQUENCES: usize = 3;
    /// Key heads of the packed cases, each a group of two query heads.
    const PACKED_KV: usize = 2;
    const PACKED_GROUP: usize = 2;
    /// Coordinates a packed key or value holds, four bits each.
    const PACKED_COORDS: usize = 32;
    const PACKED_BITS: u32 = 4;
    /// Positions a page of the packed pools holds: no multiple of any step of keys a routine walks,
    /// so a step's window straddles pages.
    const PACKED_PAGE: usize = 6;
    /// Pages each sequence's row of the table names: 300 positions, so the cubes splitting a
    /// sequence's keys each walk several steps.
    const PACKED_WIDTH: usize = 50;
    /// Pages of the packed pools: page 0 the null page, and pages no sequence names among the rest.
    const PACKED_POOL: usize = 89;
    /// A step through the pool's pages 1..PACKED_POOL coprime with their count, so consecutive
    /// pages of a sequence lie far apart and out of order.
    const PACKED_SCATTER: usize = 31;
    /// Keys each sequence attends, every one ending inside a page.
    const PACKED_LENGTHS: [u32; PACKED_SEQUENCES] = [293, 151, 40];
    /// Queries each sequence has where its queries lie end to end, the last of its keys.
    const PACKED_RUNS: [usize; PACKED_SEQUENCES] = [4, 3, 1];
    /// The norm, and the QJL residual length, of every key and value past a sequence's attended
    /// length where the cache is laid out as sequences: far from any the pool holds, so a read
    /// that does not stop at the length reads another cache than the pool.
    const PACKED_PAST_NORM: f32 = 64.;

    /// The table `[b, PACKED_WIDTH]`: each sequence's pages, as many as its attended length
    /// reaches, scattered over the pool, then the null page, page 0.
    fn packed_table() -> Vec<u32> {
        let mut next = 0;
        let mut table = Vec::new();
        for length in PACKED_LENGTHS {
            let used = (length as usize).div_ceil(PACKED_PAGE);
            for column in 0..PACKED_WIDTH {
                let page = match column < used {
                    true => (next * PACKED_SCATTER) % (PACKED_POOL - 1) + 1,
                    false => 0,
                };
                next += usize::from(column < used);
                table.push(page as u32);
            }
        }
        table
    }

    /// A pool of [`PACKED_POOL`] pages, `last` values a row, as the sequences [`packed_table`]
    /// names: a sequence's position `s` of key head `h` is its page's row up to the sequence's
    /// attended length, and `past` from there.
    fn gathered<T: Copy>(pool: &[T], last: usize, past: T) -> Vec<T> {
        let table = packed_table();
        let mut values = Vec::new();
        for (b, length) in PACKED_LENGTHS.iter().enumerate() {
            for h in 0..PACKED_KV {
                for s in 0..PACKED_WIDTH * PACKED_PAGE {
                    let page = table[b * PACKED_WIDTH + s / PACKED_PAGE] as usize;
                    let row = (page * PACKED_KV + h) * PACKED_PAGE + s % PACKED_PAGE;
                    match s < *length as usize {
                        true => values.extend_from_slice(&pool[row * last..(row + 1) * last]),
                        false => values.extend(core::iter::repeat_n(past, last)),
                    }
                }
            }
        }
        values
    }

    /// How the queries of a packed case are stated.
    #[derive(Clone, Copy, Debug)]
    enum PackedQueries {
        /// One query a sequence, a decode step's, seeing every attended key.
        Decode,
        /// [`PACKED_RUNS`] queries a sequence, end to end, causal.
        Runs,
    }

    /// That each kernel of `expected` reads a packed cache in pools of pages as it reads the same
    /// cache laid out as the sequences its table names, and that every other kernel declines the
    /// sequences. The sequences hold the pool's words, norms and, under the QJL `fixup`, residual
    /// lengths up to each sequence's attended length, and another cache past it, so only a read
    /// that stops at the length matches; the pool is scattered, its null page and the pages no
    /// sequence names filled as any other, so a read through the wrong page shows.
    ///
    /// An expected kernel that declines the sequences on this device is skipped, and says so, as is
    /// the whole case on a device without the committed plane width every packed reader runs on.
    fn packed_pages_read_as_their_sequences(
        fixup: KeyFixup,
        queries: PackedQueries,
        expected: &[AttentionKernel],
    ) {
        let client: Client = cubecl::test_device().client();
        let hw = Hardware::from_client(&client);
        if let Err(decline) = hw.plane_dim() {
            eprintln!("skipped: no packed reader runs on this device: {decline:?}");
            return;
        }
        let seq = PACKED_WIDTH * PACKED_PAGE;
        let row_words = PACKED_COORDS * PACKED_BITS as usize / 32;
        let noise = |at: usize, salt: u64| {
            (at as u64 + 1)
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9))
                >> 32
        };
        let unit = |at: usize, salt: u64| (noise(at, salt) % 4096) as f32 / 4096.;
        let pool_rows = PACKED_POOL * PACKED_KV * PACKED_PAGE;
        let words = |salt| -> Vec<u32> {
            (0..pool_rows * row_words)
                .map(|at| noise(at, salt) as u32)
                .collect()
        };
        let lengths =
            |salt| -> Vec<f32> { (0..pool_rows).map(|at| 0.5 + unit(at, salt)).collect() };
        let (key_words, key_norms, value_words, value_norms, errs) =
            (words(1), lengths(2), words(3), lengths(4), lengths(5));
        let f32s = |shape: &[usize], values: Vec<f32>| {
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                    .dtype(f32::elem_type_native())
                    .custom(values)
                    .generate_without_host_data(),
            )
        };
        let u32s = |shape: &[usize], values: &[u32]| PlainTensor::of_u32s(&client, shape, values);
        let codebook = |salt| {
            let entries = 1usize << PACKED_BITS;
            f32s(
                &[entries],
                (0..entries).map(|at| unit(at, salt) * 2. - 1.).collect(),
            )
        };
        // The cache over `rows` sequences or pages of `positions` each.
        let cache = |rows: usize,
                     positions: usize,
                     words: [Vec<u32>; 2],
                     norms: [Vec<f32>; 2],
                     errs: Vec<f32>| {
            let [key_words, value_words] = words;
            let [key_norms, value_norms] = norms;
            let words_shape = [rows, PACKED_KV, positions, row_words];
            let norms_shape = [rows, PACKED_KV, positions, 1];
            let operand = |words: &[u32], norms, salt| PackedAttentionOperand {
                words: u32s(&words_shape, words),
                norms: f32s(&norms_shape, norms),
                codebook: codebook(salt),
            };
            let fixup = match fixup {
                KeyFixup::Qjl => Some(QjlFixup {
                    errs: f32s(&norms_shape, errs),
                    sign_codebook: codebook(8),
                }),
                KeyFixup::RoundOnly => None,
            };
            PackedAttentionCache::new(
                operand(&key_words, key_norms, 6),
                operand(&value_words, value_norms, 7),
                KvBitWidths {
                    key: PACKED_BITS,
                    value: PACKED_BITS,
                },
                fixup,
            )
        };
        let in_sequences = || {
            KvSource::new(AttentionCache::Packed(cache(
                PACKED_SEQUENCES,
                seq,
                [
                    gathered(&key_words, row_words, 0),
                    gathered(&value_words, row_words, 0),
                ],
                [
                    gathered(&key_norms, 1, PACKED_PAST_NORM),
                    gathered(&value_norms, 1, PACKED_PAST_NORM),
                ],
                gathered(&errs, 1, PACKED_PAST_NORM),
            )))
        };
        let in_pages = || {
            KvSource::new(AttentionCache::Packed(cache(
                PACKED_POOL,
                PACKED_PAGE,
                [key_words.clone(), value_words.clone()],
                [key_norms.clone(), value_norms.clone()],
                errs.clone(),
            )))
            .paged(u32s(&[PACKED_SEQUENCES, PACKED_WIDTH], &packed_table()))
        };
        // The query spans the keys' lines: twice the coordinates under the fix-up.
        let dim = match fixup {
            KeyFixup::Qjl => 2 * PACKED_COORDS,
            KeyFixup::RoundOnly => PACKED_COORDS,
        };
        let heads = PACKED_KV * PACKED_GROUP;
        let f16s = |shape: &[usize]| {
            let count = shape.iter().product::<usize>();
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                    .dtype(half::f16::elem_type_native())
                    .custom((0..count).map(|at| unit(at, 9) * 2. - 1.).collect())
                    .generate_without_host_data(),
            )
        };
        let attended = || u32s(&[PACKED_SEQUENCES], &PACKED_LENGTHS);
        let (stated, visibility) = match queries {
            PackedQueries::Decode => (
                AttentionQueries::Dense {
                    query: f16s(&[PACKED_SEQUENCES, heads, 1, dim]),
                    lengths: attended(),
                },
                AttentionVisibility::Attended,
            ),
            PackedQueries::Runs => {
                let ends: Vec<u32> = PACKED_RUNS
                    .iter()
                    .scan(0, |end, &run| {
                        *end += run as u32;
                        Some(*end)
                    })
                    .collect();
                (
                    AttentionQueries::Packed {
                        query: f16s(&[PACKED_RUNS.iter().sum(), heads, dim]),
                        ends: u32s(&[PACKED_SEQUENCES], &ends),
                        longest: *PACKED_RUNS.iter().max().unwrap(),
                        lengths: attended(),
                    },
                    AttentionVisibility::Causal {
                        queries: QueryPlacement::Last,
                    },
                )
            }
        };
        let scoring = AttentionScoring::standard(visibility, PACKED_COORDS);
        let (mut served, mut failures) = (Vec::new(), Vec::new());
        for kernel in AttentionKernel::all() {
            let run = |kv: KvSource| {
                AttentionTensors::attention(&client, stated.clone(), kv, scoring)
                    .and_then(|tensors| kernel.launch_lead_on(&hw, tensors))
                    .map(|out| HostData::from_tensor_handle(&client, out, HostDataType::F32))
            };
            let reference = match (run(in_sequences()), expected.contains(&kernel)) {
                (Ok(reference), true) => reference,
                (Ok(_), false) => {
                    failures.push(format!("{kernel:?} reads a packed cache, unexpected"));
                    continue;
                }
                (Err(decline), true) => {
                    eprintln!("{kernel:?} skipped, declined on this device: {decline}");
                    continue;
                }
                (Err(decline), false) => {
                    eprintln!("{kernel:?} declines the sequences: {decline}");
                    continue;
                }
            };
            match run(in_pages()) {
                Ok(actual) => {
                    let compared =
                        cubek_test_utils::assert_equals_approx(&actual, &reference, 1e-3);
                    match compared {
                        ValidationResult::Pass => served.push(kernel),
                        compared => failures.push(format!("{kernel:?}: {compared:?}")),
                    }
                }
                Err(decline) => failures.push(format!(
                    "{kernel:?} reads the sequences, and declines their pages: {decline}"
                )),
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
        match served.is_empty() {
            true => eprintln!("skipped: this device reads the case with none of {expected:?}"),
            false => eprintln!("read the pages as their sequences: {served:?}"),
        }
    }

    /// The kernels a packed cache is read with at a decode step under round-only keys: the staged
    /// cube and both hypercube shuffles.
    const ROUND_ONLY_DECODE_READERS: [AttentionKernel; 3] = [
        AttentionKernel::Cube(CubeAttentionKernel::Fragment),
        AttentionKernel::Hypercube(HypercubeAttentionKernel::LastArrivalShuffle),
        AttentionKernel::Hypercube(HypercubeAttentionKernel::SecondLaunchShuffle),
    ];

    /// A round-only packed cache in pages reads, at a decode step, as the same cache laid out as
    /// the sequences its table names. That reference is the packed read of a cache laid out as
    /// sequences: the staged cube's tests in `cubek-attention-backward`
    /// (`tests/backward/packed.rs`) check it against the host's attention over the decoded rows;
    /// the hypercube shuffles' packed read has no host reference in this repository.
    #[test]
    fn round_only_packed_pages_read_as_their_sequences_at_a_decode_step() {
        packed_pages_read_as_their_sequences(
            KeyFixup::RoundOnly,
            PackedQueries::Decode,
            &ROUND_ONLY_DECODE_READERS,
        );
    }

    /// A packed cache whose keys carry the QJL fix-up, in pages, reads, at a decode step, as the
    /// same cache laid out as the sequences its table names, by both hypercube shuffles. That
    /// reference is the packed read of a cache laid out as sequences; its QJL read has no host
    /// reference in this repository.
    #[test]
    fn qjl_packed_pages_read_as_their_sequences_at_a_decode_step() {
        packed_pages_read_as_their_sequences(
            KeyFixup::Qjl,
            PackedQueries::Decode,
            &ROUND_ONLY_DECODE_READERS[1..],
        );
    }

    /// A round-only packed cache in pages reads, under queries that lie end to end, as the same
    /// cache laid out as the sequences its table names, by the staged cube. That reference is the
    /// staged cube's packed read of a cache laid out as sequences, which its tests in
    /// `cubek-attention-backward` (`tests/backward/packed.rs`) check against the host's attention
    /// over the decoded rows, at dense queries.
    #[test]
    fn round_only_packed_pages_read_as_their_sequences_under_query_runs() {
        packed_pages_read_as_their_sequences(
            KeyFixup::RoundOnly,
            PackedQueries::Runs,
            &ROUND_ONLY_DECODE_READERS[..1],
        );
    }
}
