//! A packed cache as the attention reads it: each operand's codebook indices packed into words,
//! its per-vector norms and the codebook, and the keys' QJL fix-up where there is one.

use crate::kernels::page_table;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::tile::kind::Field;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Dimension, Indexing, Launcher, TileSpec, TileTensorLaunch};
use cubecl::zspace::metadata::Metadata;
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{B, D, DV, KV, S};
use cubek_std::routine::Breakable;

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
/// `norms` `[b, kv, s, 1]`, or laid out as [`KvSource::pages`] states.
///
/// [`KvSource::pages`]: crate::kernels::KvSource::pages
#[derive(Debug, Clone)]
pub struct PackedAttentionOperand<T = PlainTensor> {
    pub words: T,
    pub norms: T,
    pub codebook: T,
}

/// The keys' QJL fix-up: each key's residual length `errs` `[b, kv, s, 1]`, laid out as the keys'
/// norms are, and `sign_codebook`, the codebook its indices read the residual's projected sign
/// through.
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

impl PackedAttentionCache {
    /// The cache bound under `launcher`, each operand a tile its kernel decodes as it reads it
    /// ([`Tile::decoded`](cubecl::tile::Tile::decoded)): the keys along the head dim `D`, the
    /// values along `DV`, and the keys' fix-up, the same words read through the sign codebook
    /// under the residual lengths, where they carry one. Every operand is read through `pages`
    /// where it is the cache's page table ([`KvSource::pages`](crate::kernels::KvSource::pages)).
    ///
    /// # Errors
    ///
    /// [`Decline::Broken`] for buffers other than [`PackedAttentionCache::new`] checks, and for a
    /// table that does not place the cache's pools, which
    /// [`AttentionTensors::attention`](crate::kernels::AttentionTensors::attention) checks first.
    pub(crate) fn bind(
        &self,
        launcher: &Launcher,
        pages: Option<&PlainTensor>,
    ) -> Result<PackedCacheBinding, Decline> {
        Ok(PackedCacheBinding {
            key: self.key.bind(launcher, self.bits.key, D, pages)?,
            value: self.value.bind(launcher, self.bits.value, DV, pages)?,
            fixup: self
                .fixup
                .as_ref()
                .map(|fixup| {
                    let fixup = PackedAttentionOperand {
                        words: self.key.words.clone(),
                        norms: fixup.errs.clone(),
                        codebook: fixup.sign_codebook.clone(),
                    };
                    fixup.bind(launcher, self.bits.key, D, pages)
                })
                .transpose()?,
        })
    }
}

/// A packed cache bound for a launch.
pub struct PackedCacheBinding {
    pub key: TileTensorLaunch,
    pub value: TileTensorLaunch,
    pub fixup: Option<TileTensorLaunch>,
}

/// The dimension of a codebook's table, its positions: outside every kernel's space.
const CODEBOOK: Dimension = Dimension(7);

/// What every packed binding rests on, checked by [`PackedAttentionCache::new`].
const CHECKED_BY_NEW: &str =
    "a packed cache's buffers are as PackedAttentionCache::new checks them";

impl PackedAttentionOperand {
    /// The operand bound under `launcher` along `[B, KV, S, head]`: its words' `bits`-wide fields
    /// indices into its codebook, each vector under its norm. Bound a word a line, the width a
    /// decoding read takes. The norms are paged by `pages` before the words are stated under
    /// them, as a pool's scales must be.
    fn bind(
        &self,
        launcher: &Launcher,
        bits: u32,
        head: Dimension,
        pages: Option<&PlainTensor>,
    ) -> Result<TileTensorLaunch, Decline> {
        const TABLE: ElemType = ElemType::Float(FloatKind::F32);
        let dimensions = [B, KV, S, head];
        // A norm per vector, `[b, kv, s, 1]`, scales the vector along `[b, kv, s]`.
        let binding = &self.norms.binding;
        let rows = self.norms.viewed(Metadata::new(
            binding.shape[..3].to_vec(),
            binding.strides[..3].to_vec(),
        ));
        let norms = launcher
            .arg(rows.binding, TABLE)
            .dimensions(&dimensions[..3])
            .build()
            .or_broken(CHECKED_BY_NEW)?
            .arg();
        let norms = paged(norms, pages)?;
        let codebook = TileTensorLaunch::new(
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
        let words = launcher
            .arg(values.binding, ElemType::UInt(UIntKind::U32))
            .dimensions(&dimensions)
            .vectorize(1)
            .packed(Field::Index {
                bits: bits as usize,
            })
            .build()
            .or_broken(CHECKED_BY_NEW)?
            .arg()
            .codebook(codebook)
            .or_broken(CHECKED_BY_NEW)?
            .scales(norms);
        paged(words, pages)
    }
}

/// `operand`, a buffer over `[B, KV, S, ·]`, read through [`page_table`] where there is one.
fn paged(
    operand: TileTensorLaunch,
    pages: Option<&PlainTensor>,
) -> Result<TileTensorLaunch, Decline> {
    match pages {
        Some(pages) => operand
            .indexed_covering(page_table(pages), Indexing::Coordinate(B))
            .or_broken(CHECKED_BY_ATTENTION),
        None => Ok(operand),
    }
}

/// What a packed pool's binding rests on, checked by
/// [`AttentionTensors::attention`](crate::kernels::AttentionTensors::attention).
const CHECKED_BY_ATTENTION: &str =
    "a packed pool and its table are as AttentionTensors::attention checks them";

#[cfg(test)]
mod tests {
    use crate::kernels::*;
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::zspace::Shape;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_std::Hardware;
    use cubek_test_utils::{
        HostAttention, HostData, HostDataType, HostDataVec, PackedVectors, TestInput, TestOutcome,
        ValidationResult,
    };

    const SEQUENCES_COUNT: usize = 3;
    const KV_HEADS_COUNT: usize = 2;
    /// Query heads a key head serves.
    const GROUP_HEADS_COUNT: usize = 2;
    /// Coordinates a packed key or value holds.
    const COORDS_COUNT: usize = 32;
    /// Width of a packed coordinate's field, keys and values alike.
    const FIELD_BITS: u32 = 4;
    /// Positions a page holds: no multiple of any step of keys a routine walks, so a step's window
    /// straddles pages.
    const PAGE_POSITIONS_COUNT: usize = 6;
    /// Pages a row of the table names: 300 positions, so the cubes splitting a sequence's keys
    /// each walk several steps.
    const ROW_PAGES_COUNT: usize = 50;
    /// Pages of the pool: page 0 the null page, and pages no sequence names among the rest.
    const POOL_PAGES_COUNT: usize = 89;
    /// How far apart the pool places a sequence's consecutive pages: coprime with the pool's other
    /// pages, so every one is reached once.
    const SCATTER_STRIDE_PAGES: usize = 31;
    /// Keys each sequence attends, every one ending inside a page.
    const LENGTHS: [u32; SEQUENCES_COUNT] = [293, 151, 40];
    /// Queries each sequence has where its queries lie end to end, the last of its keys.
    const RUN_QUERIES: [usize; SEQUENCES_COUNT] = [4, 3, 1];
    /// The most an output differs from the host's, its query and decoded lines rounded to `f16`.
    const TOLERANCE: f32 = 5e-3;

    const STAGED_CUBE: AttentionKernel = AttentionKernel::Cube(CubeAttentionKernel::Fragment);
    const LAST_ARRIVAL_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::LastArrivalShuffle);
    const SECOND_LAUNCH_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::SecondLaunchShuffle);
    /// The kernels that read round-only keys at a decode step.
    const ROUND_ONLY_DECODE_READERS: [AttentionKernel; 3] =
        [STAGED_CUBE, LAST_ARRIVAL_SHUFFLE, SECOND_LAUNCH_SHUFFLE];
    /// The kernels that read keys under the QJL fix-up at a decode step.
    const QJL_DECODE_READERS: [AttentionKernel; 2] = [LAST_ARRIVAL_SHUFFLE, SECOND_LAUNCH_SHUFFLE];
    /// The kernels that read round-only keys under queries that lie end to end.
    const QUERY_RUN_READERS: [AttentionKernel; 1] = [STAGED_CUBE];

    /// What a paged case reads: its keys' fix-up and how its queries are stated.
    #[derive(Clone, Copy, Debug)]
    enum PagedCaseKind {
        /// Round-only keys, one query a sequence seeing every attended key.
        RoundOnlyDecode,
        /// Keys under the QJL fix-up, one query a sequence seeing every attended key.
        QjlDecode,
        /// Round-only keys, [`RUN_QUERIES`] queries a sequence end to end, causal.
        RoundOnlyRuns,
    }

    impl PagedCaseKind {
        const ALL: [Self; 3] = [Self::RoundOnlyDecode, Self::QjlDecode, Self::RoundOnlyRuns];

        fn fixup(self) -> KeyFixup {
            match self {
                Self::QjlDecode => KeyFixup::Qjl,
                Self::RoundOnlyDecode | Self::RoundOnlyRuns => KeyFixup::RoundOnly,
            }
        }

        fn readers(self) -> &'static [AttentionKernel] {
            match self {
                Self::RoundOnlyDecode => &ROUND_ONLY_DECODE_READERS,
                Self::QjlDecode => &QJL_DECODE_READERS,
                Self::RoundOnlyRuns => &QUERY_RUN_READERS,
            }
        }

        /// Queries each sequence has.
        fn runs(self) -> [usize; SEQUENCES_COUNT] {
            match self {
                Self::RoundOnlyRuns => RUN_QUERIES,
                Self::RoundOnlyDecode | Self::QjlDecode => [1; SEQUENCES_COUNT],
            }
        }

        /// Coordinates of a query and of a key's line: twice a vector's under the fix-up.
        fn line_coords(self) -> usize {
            COORDS_COUNT * (1 + self.fixup().bits() as usize)
        }

        fn scoring(self) -> AttentionScoring {
            let visibility = match self {
                Self::RoundOnlyRuns => AttentionVisibility::Causal {
                    queries: QueryPlacement::Last,
                },
                Self::RoundOnlyDecode | Self::QjlDecode => AttentionVisibility::Attended,
            };
            AttentionScoring::standard(visibility, COORDS_COUNT)
        }

        /// Every query's coordinates, row after row of `[Σ q_b, h, d]`, rounded to `f16`.
        fn query_values(self) -> Vec<f32> {
            let count: usize = self.runs().iter().sum::<usize>()
                * KV_HEADS_COUNT
                * GROUP_HEADS_COUNT
                * self.line_coords();
            (0..count)
                .map(|at| half::f16::from_f32(unit(at, 9) * 2. - 1.).to_f32())
                .collect()
        }

        /// The queries on `client`: dense `[b, h, 1, d]` at a decode step, end to end otherwise.
        fn queries(self, client: &Client) -> AttentionQueries {
            let heads = KV_HEADS_COUNT * GROUP_HEADS_COUNT;
            let lengths = PlainTensor::of_u32s(client, &[SEQUENCES_COUNT], &LENGTHS);
            let query = |shape: &[usize]| {
                PlainTensor::of_handle(
                    TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                        .dtype(half::f16::elem_type_native())
                        .custom(self.query_values())
                        .generate_without_host_data(),
                )
            };
            let ends: Vec<u32> = self
                .runs()
                .iter()
                .scan(0, |end, &run| {
                    *end += run as u32;
                    Some(*end)
                })
                .collect();
            match self {
                Self::RoundOnlyRuns => AttentionQueries::Packed {
                    query: query(&[
                        ends[SEQUENCES_COUNT - 1] as usize,
                        heads,
                        self.line_coords(),
                    ]),
                    ends: PlainTensor::of_u32s(client, &[SEQUENCES_COUNT], &ends),
                    longest: RUN_QUERIES.into_iter().max().unwrap_or(0),
                    lengths,
                },
                Self::RoundOnlyDecode | Self::QjlDecode => AttentionQueries::Dense {
                    query: query(&[SEQUENCES_COUNT, heads, 1, self.line_coords()]),
                    lengths,
                },
            }
        }
    }

    /// A value of `[0, 1)` drawn from position `at` and `salt`, the same on every call.
    fn unit(at: usize, salt: u64) -> f32 {
        (noise(at, salt) % 4096) as f32 / 4096.
    }

    fn noise(at: usize, salt: u64) -> u64 {
        (at as u64 + 1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9))
            >> 32
    }

    /// A packed cache in a pool of [`POOL_PAGES_COUNT`] pages, on the host: every page filled,
    /// the null page and the pages no sequence names as any other, so a read through the wrong page
    /// or past a sequence's length reads values the host does not attend.
    struct PagedPool {
        pub key_words: Vec<u32>,
        pub key_norms: Vec<f32>,
        pub value_words: Vec<u32>,
        pub value_norms: Vec<f32>,
        /// The keys' residual lengths under the QJL fix-up.
        pub errs: Vec<f32>,
        pub key_codebook: Vec<f32>,
        pub value_codebook: Vec<f32>,
        pub sign_codebook: Vec<f32>,
        /// `[b, ROW_PAGES_COUNT]`: each sequence's pages, as many as its length reaches, scattered
        /// over the pool, then the null page, page 0.
        pub table: Vec<u32>,
    }

    impl PagedPool {
        fn new() -> Self {
            let vectors = POOL_PAGES_COUNT * KV_HEADS_COUNT * PAGE_POSITIONS_COUNT;
            let words_per_vector = COORDS_COUNT * FIELD_BITS as usize / 32;
            let words = |salt| -> Vec<u32> {
                (0..vectors * words_per_vector)
                    .map(|at| noise(at, salt) as u32)
                    .collect()
            };
            let norms =
                |salt| -> Vec<f32> { (0..vectors).map(|at| 0.5 + unit(at, salt)).collect() };
            let codebook = |salt| -> Vec<f32> {
                (0..1 << FIELD_BITS)
                    .map(|at| unit(at, salt) * 2. - 1.)
                    .collect()
            };
            Self {
                key_words: words(1),
                key_norms: norms(2),
                value_words: words(3),
                value_norms: norms(4),
                errs: norms(5),
                key_codebook: codebook(6),
                value_codebook: codebook(7),
                sign_codebook: codebook(8),
                table: Self::table(),
            }
        }

        fn table() -> Vec<u32> {
            let mut placed = 0;
            let mut table = Vec::with_capacity(SEQUENCES_COUNT * ROW_PAGES_COUNT);
            for length in LENGTHS {
                let used = (length as usize).div_ceil(PAGE_POSITIONS_COUNT);
                table.extend((placed..placed + used).map(|page| {
                    ((page * SCATTER_STRIDE_PAGES) % (POOL_PAGES_COUNT - 1) + 1) as u32
                }));
                table.extend(core::iter::repeat_n(0, ROW_PAGES_COUNT - used));
                placed += used;
            }
            table
        }

        /// The pool on `client`, read through its table, the keys under `fixup`.
        fn source(&self, client: &Client, fixup: KeyFixup) -> KvSource {
            let words_shape = [
                POOL_PAGES_COUNT,
                KV_HEADS_COUNT,
                PAGE_POSITIONS_COUNT,
                COORDS_COUNT * FIELD_BITS as usize / 32,
            ];
            let norms_shape = [POOL_PAGES_COUNT, KV_HEADS_COUNT, PAGE_POSITIONS_COUNT, 1];
            let f32s = |shape: &[usize], values: &[f32]| {
                PlainTensor::of_handle(
                    TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                        .dtype(f32::elem_type_native())
                        .custom(values.to_vec())
                        .generate_without_host_data(),
                )
            };
            let operand = |words: &[u32], norms: &[f32], codebook: &[f32]| PackedAttentionOperand {
                words: PlainTensor::of_u32s(client, &words_shape, words),
                norms: f32s(&norms_shape, norms),
                codebook: f32s(&[codebook.len()], codebook),
            };
            let fixup = match fixup {
                KeyFixup::Qjl => Some(QjlFixup {
                    errs: f32s(&norms_shape, &self.errs),
                    sign_codebook: f32s(&[self.sign_codebook.len()], &self.sign_codebook),
                }),
                KeyFixup::RoundOnly => None,
            };
            let cache = PackedAttentionCache::new(
                operand(&self.key_words, &self.key_norms, &self.key_codebook),
                operand(&self.value_words, &self.value_norms, &self.value_codebook),
                KvBitWidths {
                    key: FIELD_BITS,
                    value: FIELD_BITS,
                },
                fixup,
            );
            let pages =
                PlainTensor::of_u32s(client, &[SEQUENCES_COUNT, ROW_PAGES_COUNT], &self.table);
            KvSource::new(AttentionCache::Packed(cache)).paged(pages)
        }

        /// The vector of key head `kv` at position `position` of sequence `sequence`, in the pool.
        fn vector(&self, sequence: usize, kv: usize, position: usize) -> usize {
            let page = self.table[sequence * ROW_PAGES_COUNT + position / PAGE_POSITIONS_COUNT];
            (page as usize * KV_HEADS_COUNT + kv) * PAGE_POSITIONS_COUNT
                + position % PAGE_POSITIONS_COUNT
        }

        /// The attention `kind` states over the pool, as the host computes it over the decoded
        /// vectors each query sees: the output `[b, 1, h, dv]` at a decode step, `[Σ q_b, h, dv]`
        /// under queries end to end.
        fn reference(&self, kind: PagedCaseKind) -> HostData {
            let packed = |words, norms, codebook| PackedVectors {
                words,
                norms,
                codebook,
                bits: FIELD_BITS,
                coords: COORDS_COUNT,
            };
            let keys = packed(&self.key_words, &self.key_norms, &self.key_codebook);
            let fixup = packed(&self.key_words, &self.errs, &self.sign_codebook);
            let values = packed(&self.value_words, &self.value_norms, &self.value_codebook);
            let host = HostAttention {
                scale: kind.scoring().scale.value() as f64,
            };
            let (heads, line) = (KV_HEADS_COUNT * GROUP_HEADS_COUNT, kind.line_coords());
            let queries = kind.query_values();
            let mut out = Vec::new();
            let mut row = 0;
            for (sequence, (&length, run)) in LENGTHS.iter().zip(kind.runs()).enumerate() {
                for i in 0..run {
                    let seen: Vec<usize> = (0..=length as usize - run + i).collect();
                    for h in 0..heads {
                        let kv = h / GROUP_HEADS_COUNT;
                        let at = (row * heads + h) * line;
                        let query: Vec<f64> =
                            queries[at..at + line].iter().map(|&q| q as f64).collect();
                        let key = |j, d| {
                            let vector = self.vector(sequence, kv, j);
                            match d < COORDS_COUNT {
                                true => keys.decoded(vector, d),
                                false => fixup.decoded(vector, d - COORDS_COUNT),
                            }
                        };
                        let value = |j, c| values.decoded(self.vector(sequence, kv, j), c);
                        out.extend(host.attend(&query, &seen, key, value, COORDS_COUNT));
                    }
                    row += 1;
                }
            }
            let out: Vec<f32> = out.into_iter().map(|value| value as f32).collect();
            let shape: Vec<usize> = match kind {
                PagedCaseKind::RoundOnlyRuns => vec![row, heads, COORDS_COUNT],
                PagedCaseKind::RoundOnlyDecode | PagedCaseKind::QjlDecode => {
                    vec![SEQUENCES_COUNT, 1, heads, COORDS_COUNT]
                }
            };
            HostData::row_major(HostDataVec::F32(out), Shape::from(shape))
        }
    }

    /// The attention `kind` states over a paged pool, on `client`.
    fn paged_attention(client: &Client, pool: &PagedPool, kind: PagedCaseKind) -> AttentionTensors {
        AttentionTensors::attention(
            client,
            kind.queries(client),
            pool.source(client, kind.fixup()),
            kind.scoring(),
        )
        .expect("the case's tensors are verified")
    }

    /// `kernel`'s read of the paged pool `kind` states, against the host's attention over the
    /// vectors each query sees through the table. A kernel that declines on this device, and a
    /// device without the committed plane width every packed reader runs on, report a skip.
    fn check_paged_read(kind: PagedCaseKind, kernel: AttentionKernel) {
        let client: Client = cubecl::test_device().client();
        let hw = Hardware::from_client(&client);
        if let Err(decline) = hw.plane_dim() {
            let why = format!("no packed reader runs on this device: {decline:?}");
            return TestOutcome::Validated(ValidationResult::Skipped(why)).enforce();
        }
        let pool = PagedPool::new();
        match kernel.launch_lead_on(&hw, paged_attention(&client, &pool, kind)) {
            Ok(out) => cubek_test_utils::assert_equals_approx(
                &HostData::from_tensor_handle(&client, out, HostDataType::F32),
                &pool.reference(kind),
                TOLERANCE,
            )
            .as_test_outcome()
            .enforce(),
            Err(decline) => TestOutcome::Validated(ValidationResult::Skipped(format!(
                "{kernel:?} declines on this device: {decline}"
            )))
            .enforce(),
        }
    }

    #[test]
    fn the_staged_cube_reads_round_only_pages_at_a_decode_step() {
        check_paged_read(PagedCaseKind::RoundOnlyDecode, STAGED_CUBE);
    }

    #[test]
    fn the_last_arrival_shuffle_reads_round_only_pages_at_a_decode_step() {
        check_paged_read(PagedCaseKind::RoundOnlyDecode, LAST_ARRIVAL_SHUFFLE);
    }

    #[test]
    fn the_second_launch_shuffle_reads_round_only_pages_at_a_decode_step() {
        check_paged_read(PagedCaseKind::RoundOnlyDecode, SECOND_LAUNCH_SHUFFLE);
    }

    #[test]
    fn the_last_arrival_shuffle_reads_qjl_pages_at_a_decode_step() {
        check_paged_read(PagedCaseKind::QjlDecode, LAST_ARRIVAL_SHUFFLE);
    }

    #[test]
    fn the_second_launch_shuffle_reads_qjl_pages_at_a_decode_step() {
        check_paged_read(PagedCaseKind::QjlDecode, SECOND_LAUNCH_SHUFFLE);
    }

    #[test]
    fn the_staged_cube_reads_round_only_pages_under_query_runs() {
        check_paged_read(PagedCaseKind::RoundOnlyRuns, STAGED_CUBE);
    }

    /// No kernel but a case's readers reads its paged pool: each other one declines it.
    #[test]
    fn no_kernel_but_the_readers_reads_a_paged_pool() {
        let client: Client = cubecl::test_device().client();
        let hw = Hardware::from_client(&client);
        let pool = PagedPool::new();
        let read: Vec<(PagedCaseKind, AttentionKernel)> = PagedCaseKind::ALL
            .into_iter()
            .flat_map(|kind| {
                AttentionKernel::all()
                    .into_iter()
                    .map(move |kernel| (kind, kernel))
            })
            .filter(|(kind, kernel)| !kind.readers().contains(kernel))
            .filter(|&(kind, kernel)| {
                kernel
                    .launch_lead_on(&hw, paged_attention(&client, &pool, kind))
                    .is_ok()
            })
            .collect();
        assert!(
            read.is_empty(),
            "read by a kernel outside its readers: {read:?}"
        );
    }
}
