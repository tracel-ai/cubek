//! The queries of an attention, and where each user's sit among its keys.

use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::tensor::PlainTensor;

/// What an attention routine declines where the users' queries lie end to end: one that merges
/// its split keys outside the tiles that place each user's run.
pub const PACKED_QUERIES_UNREAD: Refusal = Refusal::Declined {
    why: "this attention routine reads queries as [b, h, q, d], every user the same q",
};

/// The queries of an attention's users, and each user's attended length `lengths`, `u32` `[b]`
/// on the device, or `[1]` for every user: user `b` attends its first `lengths[b]` keys, and its
/// `q_b` queries are the last `q_b` of them, at positions `lengths[b] - q_b .. lengths[b]`, causal
/// from there.
///
/// A decode step is either kind with one query a user.
#[derive(Debug, Clone)]
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, held by value as the launch's other operands are; both arms are tensors"
)]
pub enum AttentionQueries {
    /// Every user's `q` queries, `[b, h, q, d]`.
    Dense {
        query: PlainTensor,
        lengths: PlainTensor,
    },
    /// The users' queries one after another, `[Σ q_b, h, d]`: user `b`'s are
    /// `ends[b - 1] .. ends[b]` along the first dim (from `0` for the first), of the `u32`
    /// cumulative counts `ends` `[b]`. `longest` is the most queries any user has, which the
    /// launch's grid covers.
    Packed {
        query: PlainTensor,
        ends: PlainTensor,
        longest: usize,
        lengths: PlainTensor,
    },
}

#[cfg(test)]
mod tests {
    use crate::kernels::select::problem::QueryPositions;
    use crate::kernels::tune::{AttentionTuneInputs, launch_tuned};
    use crate::kernels::*;
    use cubecl::client::Client;
    use cubecl::prelude::*;
    use cubecl::std::tensor::TensorHandle;
    use cubecl::tile::launch::Refusal;
    use cubecl::zspace::Shape;
    use cubek_matmul::kernels::tensor::PlainTensor;
    use cubek_microkernel::attention::AttentionMicrokernel;
    use cubek_std::Hardware;
    use cubek_std::autotune::GroupPriority;
    use cubek_test_utils::{HostData, HostDataType, TestInput};

    /// Three users' queries, one of them a single position, and the keys each attends: user `b`'s
    /// queries are the last `QUERIES[b]` of its first `LENGTHS[b]` keys. The longest spans several
    /// boxes of queries, so the shorter users' later boxes lie past their runs.
    const QUERIES: [usize; 3] = [5, 1, 40];
    const LENGTHS: [usize; 3] = [12, 7, 48];
    const HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const KEYS: usize = 64;
    const HEAD_DIM: usize = 32;

    /// How a check launches one attention: its output, or `None` where it declines.
    type Launch<'a> = &'a dyn Fn(&Client, AttentionQueries, KvSource) -> Option<TensorHandle>;

    /// Every routine that reads queries lying end to end, the cube routines, and the autotune of
    /// many query positions, writes each user's queries as one launch of that user alone does, under the
    /// causal mask placed at the user's own length.
    #[test]
    fn packed_queries_match_a_launch_per_user() {
        let failures: Vec<String> = [
            check("cube staged", &|client, queries, kv| {
                lead(CubeStagedAttentionRoutine, client, queries, kv)
            }),
            check("cube shuffle", &|client, queries, kv| {
                lead(CubeShuffleAttentionRoutine, client, queries, kv)
            }),
            check("autotune", &|client, queries, kv| {
                let inputs = AttentionTuneInputs::attention(client, queries, kv, None, scoring())
                    .expect("the case's tensors are verified");
                let device = cubecl::test_device();
                Some(launch_tuned(
                    QueryPositions::Many,
                    &device,
                    inputs,
                    GroupPriority::DEFAULT,
                ))
            }),
        ]
        .into_iter()
        .flatten()
        .collect();
        assert!(failures.is_empty(), "{failures:#?}");
    }

    fn scoring() -> AttentionScoring {
        AttentionScoring::standard(
            AttentionVisibility::new(true, None, QueryPlacement::Last),
            HEAD_DIM,
        )
    }

    /// `routine` at its lead, or `None` where it declines on this device.
    fn lead<M: AttentionMicrokernel, SM, R: AttentionRoutine<M, SM>>(
        routine: R,
        client: &Client,
        queries: AttentionQueries,
        kv: KvSource,
    ) -> Option<TensorHandle> {
        let tensors = AttentionTensors::attention(client, queries, kv, scoring())
            .expect("the case's tensors are verified");
        routine
            .launch_lead_on(&Hardware::from_client(client), tensors)
            .map_err(|refusal: Refusal| eprintln!("declined on this device: {refusal:?}"))
            .ok()
    }

    /// Where `launch` writes another value for packed queries than for a launch per user, the
    /// first such value.
    fn check(name: &str, launch: Launch) -> Option<String> {
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
        let per_key = KV_HEADS * KEYS * HEAD_DIM;
        let per_query = HEADS * HEAD_DIM;
        let total: usize = QUERIES.iter().sum();
        let (queries, keys, cached_values) = (
            values(1, total * per_query),
            values(2, 3 * per_key),
            values(3, 3 * per_key),
        );
        let float = |key: Vec<f32>, value: Vec<f32>, users: usize| {
            let shape = [users, KV_HEADS, KEYS, HEAD_DIM];
            KvSource::new(AttentionCache::Float {
                key: tensor(&shape, f16, key),
                value: tensor(&shape, f16, value),
            })
        };
        let ends: Vec<f32> = QUERIES
            .iter()
            .scan(0, |end, &queries| {
                *end += queries;
                Some(*end as f32)
            })
            .collect();
        let packed = AttentionQueries::Packed {
            query: tensor(&[total, HEADS, HEAD_DIM], f16, queries.clone()),
            ends: tensor(&[3], u32, ends),
            longest: *QUERIES.iter().max().unwrap(),
            lengths: tensor(&[3], u32, LENGTHS.map(|length| length as f32).to_vec()),
        };
        let kv = float(keys.clone(), cached_values.clone(), 3);
        let out = launch(&client, packed, kv)?;
        let packed = HostData::from_tensor_handle(&client, out, HostDataType::F32);

        let mut start = 0;
        for (user, &count) in QUERIES.iter().enumerate() {
            // The user's queries `[1, h, q, d]` from its run of `[Σ q, h, d]`.
            let mut own = Vec::with_capacity(count * per_query);
            for head in 0..HEADS {
                for query in 0..count {
                    let at = ((start + query) * HEADS + head) * HEAD_DIM;
                    own.extend_from_slice(&queries[at..at + HEAD_DIM]);
                }
            }
            let user_keys = |all: &[f32]| all[user * per_key..(user + 1) * per_key].to_vec();
            let dense = AttentionQueries::Dense {
                query: tensor(&[1, HEADS, count, HEAD_DIM], f16, own),
                lengths: tensor(&[1], u32, vec![LENGTHS[user] as f32]),
            };
            let kv = float(user_keys(&keys), user_keys(&cached_values), 1);
            let out =
                launch(&client, dense, kv).expect("what reads packed queries reads dense ones");
            let dense = HostData::from_tensor_handle(&client, out, HostDataType::F32);
            for query in 0..count {
                for head in 0..HEADS {
                    for c in 0..HEAD_DIM {
                        let want = dense.get_f32(&[0, query, head, c]);
                        let got = packed.get_f32(&[start + query, head, c]);
                        if (got - want).abs() > 5e-3 * (1. + want.abs()) {
                            return Some(format!(
                                "{name}: user {user} query {query} head {head} [{c}]: {got}, \
                                 expected {want}"
                            ));
                        }
                    }
                }
            }
            start += count;
        }
        None
    }
}
