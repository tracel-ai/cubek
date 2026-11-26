//! The queries of an attention, and where each sequence's sit among its keys.

use cubecl::tile::launch::Decline;
use cubek_matmul::kernels::tensor::PlainTensor;

/// What an attention routine declines where the sequences' queries lie end to end: one that merges
/// its split keys outside the tiles that place each sequence's run.
pub const PACKED_QUERIES_UNREAD: Decline = Decline::Unserved {
    why: "this attention routine reads queries as [b, h, q, d], every sequence the same q",
};

/// The queries of an attention's sequences, and each sequence's attended length `lengths`, `u32`
/// `[b]` on the device, or `[1]` for every sequence: sequence `b` attends its first `lengths[b]`
/// keys, and its `q_b` queries are the last `q_b` of them, at positions `lengths[b] - q_b ..
/// lengths[b]`, causal from there.
///
/// A decode step is either kind with one query a sequence.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, kept by value as the launch's other operands are; both arms are tensors"
)]
pub enum AttentionQueries {
    /// Every sequence's `q` queries, `[b, h, q, d]`.
    Dense {
        query: PlainTensor,
        lengths: PlainTensor,
    },
    /// The sequences' queries one after another, `[Σ q_b, h, d]`: sequence `b`'s are
    /// `ends[b - 1] .. ends[b]` along the first dim (from `0` for the first), of the `u32`
    /// cumulative counts `ends` `[b]`. `longest` is the most queries any sequence has, which the
    /// launch's grid covers.
    Packed {
        query: PlainTensor,
        ends: PlainTensor,
        longest: usize,
        lengths: PlainTensor,
    },
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::kernels::*;
    use core::num::NonZeroUsize;
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::std::tensor::TensorHandle;
    use cubecl::tile::launch::Decline;
    use cubecl::zspace::Shape;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_std::Hardware;
    use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};

    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const KEYS: usize = 64;
    const HEAD_DIM: usize = 32;
    /// Positions a page holds, where a case reads its keys from pools of pages.
    const PAGE_SIZE: usize = 16;

    /// Sequences' queries and the keys each attends: sequence `b`'s queries are the last
    /// `queries[b]` of its first `lengths[b]` keys.
    struct Case {
        name: &'static str,
        queries: &'static [usize],
        lengths: &'static [usize],
        /// Every sequence's length bound as one `[1]` length, at stride zero.
        one_length: bool,
        window: Option<usize>,
        paged: bool,
    }

    const BASE: Case = Case {
        name: "three sequences",
        queries: &[5, 1, 40],
        lengths: &[12, 7, 48],
        one_length: false,
        window: None,
        paged: false,
    };

    /// The longest spans several boxes of queries, so the shorter sequences' later boxes lie past
    /// their runs; a sequence with no queries holds an empty run; one sequence is a run alone.
    const CASES: [Case; 6] = [
        BASE,
        Case {
            name: "a sequence with no queries",
            queries: &[5, 0, 40],
            ..BASE
        },
        Case {
            name: "one sequence",
            queries: &[40],
            lengths: &[48],
            ..BASE
        },
        Case {
            name: "one length for every sequence",
            queries: &[5, 1, 40],
            lengths: &[48, 48, 48],
            one_length: true,
            ..BASE
        },
        Case {
            name: "a window",
            window: Some(8),
            ..BASE
        },
        Case {
            name: "keys in pages",
            paged: true,
            ..BASE
        },
    ];

    /// How a check launches one attention: its output, or `None` where it declines.
    type Launch<'a> =
        &'a dyn Fn(&Client, AttentionQueries, KvSource, AttentionScoring) -> Option<TensorHandle>;

    /// The cube routines, the only ones that read packed queries, write each sequence's queries
    /// as one launch of that sequence alone does, under the causal mask placed at the sequence's
    /// own length, in every case.
    #[test]
    fn packed_queries_match_a_launch_per_sequence() {
        let launches: [(&str, Launch); 2] = [
            ("cube staged", &|client, queries, kv, scoring| {
                lead(
                    DirectAttention(CubeStagedAttentionRoutine),
                    client,
                    queries,
                    kv,
                    scoring,
                )
            }),
            ("cube shuffle", &|client, queries, kv, scoring| {
                lead(
                    DirectAttention(CubeShuffleAttentionRoutine),
                    client,
                    queries,
                    kv,
                    scoring,
                )
            }),
        ];
        let failures: Vec<String> = CASES
            .iter()
            .flat_map(|case| {
                launches
                    .iter()
                    .filter_map(move |(name, launch)| check(case, name, *launch))
            })
            .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// `launch` at its lead, or `None` where it declines on this device, which reports a skip. A
    /// broken launch fails.
    fn lead(
        launch: impl AttentionLaunch,
        client: &Client,
        queries: AttentionQueries,
        kv: KvSource,
        scoring: AttentionScoring,
    ) -> Option<TensorHandle> {
        let tensors = AttentionTensors::attention(client, queries, kv, scoring)
            .expect("the case's tensors are verified");
        match launch.launch_lead_on(&Hardware::from_client(client), tensors) {
            Ok(out) => Some(out),
            Err(broken) if broken.is_broken() => panic!("{broken}"),
            Err(decline) => {
                let why = format!("declined on this device: {decline}");
                TestOutcome::Validated(ValidationResult::Skipped(why)).enforce();
                None
            }
        }
    }

    /// Where `launch` writes another value for `case`'s packed queries than for a launch per
    /// sequence, the first such value.
    fn check(case: &Case, name: &str, launch: Launch) -> Option<String> {
        let client: Client = cubecl::test_device().client();
        let (f16, u32) = (half::f16::elem_type_native(), u32::elem_type_native());
        let tensor = |shape: &[usize], element, data: Vec<f32>| {
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                    .dtype(element)
                    .custom(data)
                    .generate_without_host_data(),
            )
        };
        // Values a few hundredths apart, so the softmax weighs every key differently.
        let values = |seed: usize, count: usize| -> Vec<f32> {
            (0..count)
                .map(|i| (((i * 7919 + seed * 104_729) % 97) as f32 - 48.) / 64.)
                .collect()
        };
        let scoring = AttentionScoring::standard(
            AttentionVisibility::new(
                true,
                case.window.and_then(NonZeroUsize::new),
                QueryPlacement::Last,
            ),
            HEAD_DIM,
        );
        let sequences = case.queries.len();
        let per_key = KV_HEADS * KEYS * HEAD_DIM;
        let per_query = HEADS * HEAD_DIM;
        let total: usize = case.queries.iter().sum();
        let (queries, keys, cached_values) = (
            values(1, total * per_query),
            values(2, sequences * per_key),
            values(3, sequences * per_key),
        );
        // `[b, kv, s, d]` from `[b, kv, s, d]` values, or the same keys in pools of pages, each
        // sequence's pages in order, read through a table of their own.
        let cache = |key: &[f32], value: &[f32], count: usize| {
            if !case.paged {
                let shape = [count, KV_HEADS, KEYS, HEAD_DIM];
                return KvSource::new(AttentionCache::Float {
                    key: tensor(&shape, f16, key.to_vec()),
                    value: tensor(&shape, f16, value.to_vec()),
                });
            }
            let per_sequence = KEYS / PAGE_SIZE;
            let pooled = |values: &[f32]| -> Vec<f32> {
                let mut pool = Vec::with_capacity(values.len());
                for page in 0..count * per_sequence {
                    let (sequence, first) = (page / per_sequence, page % per_sequence * PAGE_SIZE);
                    for head in 0..KV_HEADS {
                        for position in first..first + PAGE_SIZE {
                            let at = ((sequence * KV_HEADS + head) * KEYS + position) * HEAD_DIM;
                            pool.extend_from_slice(&values[at..at + HEAD_DIM]);
                        }
                    }
                }
                pool
            };
            let shape = [count * per_sequence, KV_HEADS, PAGE_SIZE, HEAD_DIM];
            let table: Vec<f32> = (0..count * per_sequence).map(|page| page as f32).collect();
            KvSource::new(AttentionCache::Float {
                key: tensor(&shape, f16, pooled(key)),
                value: tensor(&shape, f16, pooled(value)),
            })
            .paged(tensor(&[count, per_sequence], u32, table))
        };
        let lengths = |lengths: &[usize]| {
            let lengths = match case.one_length {
                true => &lengths[..1],
                false => lengths,
            };
            let values = lengths.iter().map(|&length| length as f32).collect();
            tensor(&[lengths.len()], u32, values)
        };
        let ends: Vec<f32> = case
            .queries
            .iter()
            .scan(0, |end, &queries| {
                *end += queries;
                Some(*end as f32)
            })
            .collect();
        let packed = AttentionQueries::Packed {
            query: tensor(&[total, HEADS, HEAD_DIM], f16, queries.clone()),
            ends: tensor(&[sequences], u32, ends),
            longest: *case.queries.iter().max().unwrap(),
            lengths: lengths(case.lengths),
        };
        let out = launch(
            &client,
            packed,
            cache(&keys, &cached_values, sequences),
            scoring,
        )?;
        let packed = HostData::from_tensor_handle(&client, out, HostDataType::F32);

        let mut start = 0;
        for (sequence, &count) in case.queries.iter().enumerate() {
            // A sequence with no queries writes nothing to compare.
            if count == 0 {
                continue;
            }
            // The sequence's queries `[1, h, q, d]` from its run of `[Σ q, h, d]`.
            let mut own = Vec::with_capacity(count * per_query);
            for head in 0..HEADS {
                for query in 0..count {
                    let at = ((start + query) * HEADS + head) * HEAD_DIM;
                    own.extend_from_slice(&queries[at..at + HEAD_DIM]);
                }
            }
            let own_keys = |all: &[f32]| all[sequence * per_key..(sequence + 1) * per_key].to_vec();
            let dense = AttentionQueries::Dense {
                query: tensor(&[1, HEADS, count, HEAD_DIM], f16, own),
                lengths: lengths(&case.lengths[sequence..=sequence]),
            };
            let kv = cache(&own_keys(&keys), &own_keys(&cached_values), 1);
            let out = launch(&client, dense, kv, scoring)
                .expect("what reads packed queries reads dense ones");
            let dense = HostData::from_tensor_handle(&client, out, HostDataType::F32);
            for query in 0..count {
                for head in 0..HEADS {
                    for c in 0..HEAD_DIM {
                        let want = dense.get_f32(&[0, query, head, c]);
                        let got = packed.get_f32(&[start + query, head, c]);
                        if (got - want).abs() > 5e-3 * (1. + want.abs()) {
                            return Some(format!(
                                "{}, {name}: sequence {sequence} query {query} head {head} \
                                 [{c}]: {got}, expected {want}",
                                case.name
                            ));
                        }
                    }
                }
            }
            start += count;
        }
        None
    }

    /// Many sequences of one query at Devstral Small 2's heads, 32 of 128 over 8 key heads,
    /// packed in one launch, write what the same sequences laid out as a batch do. The packed
    /// query and output, viewed once per sequence with the batch at stride zero, are `2^36`
    /// values as their logical shapes count them, past a `u32` whatever the width they are bound
    /// at, against `2^24` in their buffers: the launch counts what the buffers hold.
    #[test]
    fn many_packed_sequences_match_a_batch() {
        const SEQUENCES: usize = 4096;
        const HEADS: usize = 32;
        const KV_HEADS: usize = 8;
        const HEAD_DIM: usize = 128;
        /// Keys a page holds, every sequence's in one page of a pool of `POOL`.
        const PAGE_SIZE: usize = 16;
        const POOL: usize = 4;

        let client: Client = cubecl::test_device().client();
        if let Err(refusal) = Hardware::from_client(&client).plane_dim() {
            let why = format!("no plane on this device: {refusal}");
            return TestOutcome::Validated(ValidationResult::Skipped(why)).enforce();
        }
        let (f16, u32) = (half::f16::elem_type_native(), u32::elem_type_native());
        let tensor = |shape: &[usize], element, data: Vec<f32>| {
            PlainTensor::of_handle(
                TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                    .dtype(element)
                    .custom(data)
                    .generate_without_host_data(),
            )
        };
        let values = |seed: usize, count: usize| -> Vec<f32> {
            (0..count)
                .map(|i| (((i * 7919 + seed * 104_729) % 97) as f32 - 48.) / 64.)
                .collect()
        };
        let queries = values(1, SEQUENCES * HEADS * HEAD_DIM);
        let pool = [POOL, KV_HEADS, PAGE_SIZE, HEAD_DIM];
        let (keys, cached_values) = (
            values(2, pool.iter().product()),
            values(3, pool.iter().product()),
        );
        let table: Vec<f32> = (0..SEQUENCES).map(|b| (b % POOL) as f32).collect();
        // Each sequence attends a length of its own, so a run read at another sequence's place
        // writes another value.
        let lengths: Vec<f32> = (0..SEQUENCES).map(|b| (1 + b % PAGE_SIZE) as f32).collect();
        let cache = || {
            KvSource::new(AttentionCache::Float {
                key: tensor(&pool, f16, keys.clone()),
                value: tensor(&pool, f16, cached_values.clone()),
            })
            .paged(tensor(&[SEQUENCES, 1], u32, table.clone()))
        };
        let scoring = AttentionScoring::standard(
            AttentionVisibility::new(true, None, QueryPlacement::Last),
            HEAD_DIM,
        );
        let ends: Vec<f32> = (1..=SEQUENCES).map(|end| end as f32).collect();
        let packed = || AttentionQueries::Packed {
            query: tensor(&[SEQUENCES, HEADS, HEAD_DIM], f16, queries.clone()),
            ends: tensor(&[SEQUENCES], u32, ends.clone()),
            longest: 1,
            lengths: tensor(&[SEQUENCES], u32, lengths.clone()),
        };
        // `[Σ q, h, d]` at one query a sequence is `[b, h, 1, d]` in the same order.
        let batch = || AttentionQueries::Dense {
            query: tensor(&[SEQUENCES, HEADS, 1, HEAD_DIM], f16, queries.clone()),
            lengths: tensor(&[SEQUENCES], u32, lengths.clone()),
        };
        type Routine<'a> = &'a dyn Fn(AttentionQueries) -> Option<TensorHandle>;
        let routines: [(&str, Routine); 2] = [
            ("cube staged", &|queries| {
                lead(
                    DirectAttention(CubeStagedAttentionRoutine),
                    &client,
                    queries,
                    cache(),
                    scoring,
                )
            }),
            ("cube shuffle", &|queries| {
                lead(
                    DirectAttention(CubeShuffleAttentionRoutine),
                    &client,
                    queries,
                    cache(),
                    scoring,
                )
            }),
        ];
        for (name, launch) in routines {
            let Some(packed) = launch(packed()) else {
                continue;
            };
            let batch = launch(batch()).expect("what reads packed queries reads dense ones");
            let packed = HostData::from_tensor_handle(&client, packed, HostDataType::F32);
            let batch = HostData::from_tensor_handle(&client, batch, HostDataType::F32);
            for sequence in 0..SEQUENCES {
                for head in 0..HEADS {
                    for c in 0..HEAD_DIM {
                        let want = batch.get_f32(&[sequence, 0, head, c]);
                        let got = packed.get_f32(&[sequence, head, c]);
                        assert!(
                            (got - want).abs() <= 5e-3 * (1. + want.abs()),
                            "{name}: sequence {sequence} head {head} [{c}]: {got}, expected {want}"
                        );
                    }
                }
            }
        }
    }

    /// A length past the keys attends as a length of the keys there are, on every kernel at its
    /// lead and on the hypercube stream routine: one query seeing every key, and several queries
    /// under a causal visibility, which the length places. The key head after another's keys in the
    /// buffer is what a read past them would reach.
    #[test]
    fn a_length_past_the_keys_attends_the_keys() {
        const LENGTH_PAST_THE_KEYS: u32 = 3 * KEYS as u32;
        let client: Client = cubecl::test_device().client();
        let cache_shape = [1, KV_HEADS, KEYS, HEAD_DIM];
        let cases = [
            (1, AttentionVisibility::Attended),
            (
                5,
                AttentionVisibility::new(true, None, QueryPlacement::Last),
            ),
        ];
        assert_past_reads_as_within(
            &every_launch(),
            &cases,
            |launch, (queries, visibility), past| {
                let length = match past {
                    true => LENGTH_PAST_THE_KEYS,
                    false => KEYS as u32,
                };
                let tensors = AttentionTensors::new(
                    &client,
                    random(&client, &[1, HEADS, queries, HEAD_DIM], 1),
                    random(&client, &cache_shape, 2),
                    random(&client, &cache_shape, 3),
                    PlainTensor::of_u32s(&client, &[1], &[length]),
                    AttentionScoring::standard(visibility, HEAD_DIM),
                )
                .expect("the case's tensors are verified");
                launch(&Hardware::from_client(&client), tensors)
                    .map(|out| HostData::from_tensor_handle(&client, out, HostDataType::F32))
            },
        );
    }

    /// An entry of a page table past the pool reads the pool's last page, on every kernel that
    /// reads pages: the same as an entry naming that page.
    #[test]
    fn a_page_past_the_pool_reads_its_last_page() {
        const POOL: usize = KEYS / PAGE_SIZE;
        const PAGE_PAST_THE_POOL: u32 = 100 * POOL as u32;
        let client: Client = cubecl::test_device().client();
        let pool_shape = [POOL, KV_HEADS, PAGE_SIZE, HEAD_DIM];
        let cases = [AttentionVisibility::Attended];
        assert_past_reads_as_within(&every_launch(), &cases, |launch, visibility, past| {
            let mut table: Vec<u32> = (0..POOL as u32).collect();
            if past {
                table[POOL - 1] = PAGE_PAST_THE_POOL;
            }
            let tensors = AttentionTensors::paged(
                &client,
                random(&client, &[1, HEADS, 1, HEAD_DIM], 1),
                random(&client, &pool_shape, 2),
                random(&client, &pool_shape, 3),
                PlainTensor::of_u32s(&client, &[1, POOL], &table),
                PlainTensor::of_u32s(&client, &[1], &[KEYS as u32]),
                AttentionScoring::standard(visibility, HEAD_DIM),
            )
            .expect("the case's tensors are verified");
            launch(&Hardware::from_client(&client), tensors)
                .map(|out| HostData::from_tensor_handle(&client, out, HostDataType::F32))
        });
    }

    /// A listed block past its sequence's pages lies in their last page, and a page of the table
    /// past the pool is its last page, on every kernel that reads listed blocks: the same as the
    /// block or the page in range there.
    #[test]
    fn a_listed_block_past_the_pages_reads_the_last_page() {
        const BLOCK: usize = 16;
        const BLOCKS: u32 = (KEYS / BLOCK) as u32;
        let client: Client = cubecl::test_device().client();
        let cache_shape = [1, KV_HEADS, KEYS, HEAD_DIM];
        let cases = ["a block past the pages", "a page past the pool"];
        assert_past_reads_as_within(&every_launch(), &cases, |launch, case, past| {
            let mut blocks: Vec<u32> = (0..BLOCKS).collect();
            let mut page = 0;
            match (case, past) {
                ("a block past the pages", true) => blocks[0] = 1000 * BLOCKS + BLOCKS - 1,
                ("a block past the pages", false) => blocks[0] = BLOCKS - 1,
                (_, true) => page = 100,
                (_, false) => {}
            }
            let kv = KvSource::new(AttentionCache::Float {
                key: random(&client, &cache_shape, 2),
                value: random(&client, &cache_shape, 3),
            })
            .paged(PlainTensor::of_u32s(&client, &[1, 1], &[page]))
            .indexed(
                PlainTensor::of_u32s(&client, &[1, BLOCKS as usize], &blocks),
                NonZeroUsize::new(BLOCK).expect("a block holds keys"),
            );
            let queries = AttentionQueries::Dense {
                query: random(&client, &[1, HEADS, 1, HEAD_DIM], 1),
                lengths: PlainTensor::of_u32s(&client, &[1], &[KEYS as u32]),
            };
            let scoring = AttentionScoring::standard(AttentionVisibility::Attended, HEAD_DIM);
            let tensors = AttentionTensors::attention(&client, queries, kv, scoring)
                .expect("the case's tensors are verified");
            launch(&Hardware::from_client(&client), tensors)
                .map(|out| HostData::from_tensor_handle(&client, out, HostDataType::F32))
        });
    }

    /// Queries lying end to end whose last end is past their count read and write no query past
    /// them, and count no query past them: every query there is is written as under ends that stop
    /// at the count, all seeing every key and under a causal visibility the count places, by the
    /// cube routines, the only ones that read queries end to end.
    #[test]
    fn query_ends_past_the_queries_read_the_queries_there_are() {
        const RUNS: [u32; 2] = [3, 2];
        const END_PAST_THE_QUERIES: u32 = 1000;
        let client: Client = cubecl::test_device().client();
        let count = RUNS.iter().sum::<u32>() as usize;
        let cache_shape = [RUNS.len(), KV_HEADS, KEYS, HEAD_DIM];
        let launches: Vec<(String, LeadLaunch)> = vec![
            (
                "cube staged".to_string(),
                Box::new(|hw, tensors| {
                    DirectAttention(CubeStagedAttentionRoutine).launch_lead_on(hw, tensors)
                }),
            ),
            (
                "cube shuffle".to_string(),
                Box::new(|hw, tensors| {
                    DirectAttention(CubeShuffleAttentionRoutine).launch_lead_on(hw, tensors)
                }),
            ),
        ];
        let cases = [
            AttentionVisibility::Attended,
            AttentionVisibility::new(true, None, QueryPlacement::Last),
        ];
        assert_past_reads_as_within(&launches, &cases, |launch, visibility, past| {
            let last = match past {
                true => END_PAST_THE_QUERIES,
                false => count as u32,
            };
            let queries = AttentionQueries::Packed {
                query: random(&client, &[count, HEADS, HEAD_DIM], 1),
                ends: PlainTensor::of_u32s(&client, &[RUNS.len()], &[RUNS[0], last]),
                longest: RUNS[0] as usize,
                lengths: PlainTensor::of_u32s(&client, &[1], &[KEYS as u32]),
            };
            let kv = KvSource::new(AttentionCache::Float {
                key: random(&client, &cache_shape, 2),
                value: random(&client, &cache_shape, 3),
            });
            let tensors = AttentionTensors::attention(
                &client,
                queries,
                kv,
                AttentionScoring::standard(visibility, HEAD_DIM),
            )
            .expect("the case's tensors are verified");
            launch(&Hardware::from_client(&client), tensors)
                .map(|out| HostData::from_tensor_handle(&client, out, HostDataType::F32))
        });
    }

    /// A launch of the tensors at its lead on a device.
    pub(crate) type LeadLaunch =
        Box<dyn Fn(&Hardware, AttentionTensors) -> Result<TensorHandle, Decline>>;

    /// Every kernel at its lead, and the hypercube stream routine, which no kernel names.
    fn every_launch() -> Vec<(String, LeadLaunch)> {
        let mut launches: Vec<(String, LeadLaunch)> = AttentionKernel::all()
            .into_iter()
            .map(|kernel| {
                let launch: LeadLaunch =
                    Box::new(move |hw, tensors| kernel.launch_lead_on(hw, tensors));
                (format!("{kernel:?}"), launch)
            })
            .collect();
        launches.push((
            "hypercube stream".to_string(),
            Box::new(|hw, tensors| {
                DirectAttention(HypercubeStreamFragmentAttentionRoutine).launch_lead_on(hw, tensors)
            }),
        ));
        launches
    }

    /// `f32` values of `shape` drawn uniformly from `[-1, 1)` under `seed`, on `client`.
    fn random(client: &Client, shape: &[usize], seed: u64) -> PlainTensor {
        PlainTensor::of_handle(
            TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                .dtype(f32::elem_type_native())
                .uniform(seed, -1.0, 1.0)
                .generate_without_host_data(),
        )
    }

    /// Each of `launches` on each of `cases`, `attend` writing the attention of its operands with a
    /// value past its range where told so, against the same within range. A launch that declines
    /// on this device is a skip, and a broken one fails.
    pub(crate) fn assert_past_reads_as_within<C: core::fmt::Debug + Copy>(
        launches: &[(String, LeadLaunch)],
        cases: &[C],
        attend: impl Fn(&LeadLaunch, C, bool) -> Result<HostData, Decline>,
    ) {
        let (mut wrong, mut declined) = (Vec::new(), Vec::new());
        for (name, launch) in launches {
            for &case in cases {
                let case_name = format!("{name}, {case:?}");
                match (attend(launch, case, false), attend(launch, case, true)) {
                    (Ok(expected), Ok(actual)) => {
                        let checked =
                            cubek_test_utils::assert_equals_approx(&actual, &expected, 1e-4);
                        if !matches!(checked, ValidationResult::Pass) {
                            wrong.push(format!("{case_name}: {checked:?}"));
                        }
                    }
                    (Err(decline), _) | (_, Err(decline)) => declined.push((case_name, decline)),
                }
            }
        }
        let broken: Vec<_> = declined.iter().filter(|(_, why)| why.is_broken()).collect();
        assert!(broken.is_empty(), "{broken:#?}");
        assert!(wrong.is_empty(), "{wrong:#?}");
        for (case, why) in declined {
            TestOutcome::Validated(ValidationResult::Skipped(format!("{case}: {why}"))).enforce();
        }
    }
}
