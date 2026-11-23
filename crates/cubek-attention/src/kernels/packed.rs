//! A packed cache as the attention reads it: each operand's codebook indices packed into words,
//! its per-vector norms and the codebook, and the keys' QJL fix-up where there is one.

use crate::kernels::page_table;
use cubecl::ir::{ElemType, FloatKind, UIntKind};
use cubecl::tile::kind::Field;
use cubecl::tile::{Dimension, Indexing, Launcher, TileSpec, TileTensorLaunch};
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
/// `norms` `[b, kv, s, 1]`. Where the cache lies in pools of pages ([`KvSource::pages`]),
/// `words` and `norms` are `[pages, kv, page_size, ·]`, both placed by the one table.
///
/// [`KvSource::pages`]: crate::kernels::KvSource::pages
#[derive(Debug, Clone)]
pub struct PackedAttentionOperand<T = PlainTensor> {
    pub words: T,
    pub norms: T,
    pub codebook: T,
}

/// The keys' QJL fix-up: each key's residual length `errs` `[b, kv, s, 1]`, a pool
/// `[pages, kv, page_size, 1]` as the keys' words are, and `sign_codebook`, the codebook its
/// indices read the residual's projected sign through.
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
    /// under the residual lengths, where they carry one. Where `pages` is the table of a pool of
    /// pages ([`KvSource::pages`](crate::kernels::KvSource::pages)), every operand's words and
    /// norms are read through it, as a float pool's values are.
    pub fn bind(&self, launcher: &Launcher, pages: Option<&PlainTensor>) -> PackedCacheBinding {
        PackedCacheBinding {
            key: self.key.bind(launcher, self.bits.key, D, pages),
            value: self.value.bind(launcher, self.bits.value, DV, pages),
            fixup: self.fixup.as_ref().map(|fixup| {
                let fixup = PackedAttentionOperand {
                    words: self.key.words.clone(),
                    norms: fixup.errs.clone(),
                    codebook: fixup.sign_codebook.clone(),
                };
                fixup.bind(launcher, self.bits.key, D, pages)
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
    /// decoding read takes. Where `pages` places a pool, the norms are a pool placed by it before
    /// they are stated, as a pool's scales must be, and the words are placed under them.
    fn bind(
        &self,
        launcher: &Launcher,
        bits: u32,
        head: Dimension,
        pages: Option<&PlainTensor>,
    ) -> TileTensorLaunch {
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
            .expect(CHECKED_BY_NEW)
            .arg();
        let norms = paged(norms, pages);
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
            .expect(CHECKED_BY_NEW)
            .arg()
            .codebook(codebook)
            .expect(CHECKED_BY_NEW)
            .scales(norms);
        paged(words, pages)
    }
}

/// `operand`, a buffer over `[B, KV, S, ·]`, read as a pool of pages through `pages` where there
/// is one: the table covers the launch, a row a sequence and a column a page of its positions.
fn paged(operand: TileTensorLaunch, pages: Option<&PlainTensor>) -> TileTensorLaunch {
    match pages {
        Some(pages) => operand
            .indexed_covering(page_table(pages), Indexing::Coordinate(B))
            .expect(CHECKED_BY_ATTENTION),
        None => operand,
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
    use cubek_test_utils::{HostData, HostDataType, TestInput, ValidationResult};

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
