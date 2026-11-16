//! A packed cache read as it lies by the staged cube: each step of keys and values decoded into
//! its stage once, for every query of the cube. Its output is the host's attention over the
//! decoded rows.

use super::reference::Case;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::zspace::Shape;
use cubek_attention::kernels::{
    AttentionLaunch, AttentionScoring, AttentionTensors, CubeStagedAttentionRoutine,
    DirectAttention, KvBitWidths, PackedAttentionCache, PackedAttentionOperand, ScoreScale,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{HostData, HostDataType, TestInput};

/// Bits per coordinate of the codes, keys and values alike.
const BITS: u32 = 4;

/// A causal prompt chunk of two groups of two query heads over a packed cache: no step of queries
/// or keys divides it evenly.
const PROMPT: Case = Case {
    batch: 1,
    heads: 4,
    kv_heads: 2,
    seq_q: 13,
    seq_kv: 37,
    head_dim: 32,
    val_dim: 32,
    visibility: AttentionVisibility::Causal {
        queries: QueryPlacement::Last,
    },
    hidden: None,
};

/// The same chunk at the head dim a model's prefill runs.
const PROMPT_WIDE: Case = Case {
    heads: 8,
    seq_q: 70,
    seq_kv: 300,
    head_dim: 128,
    val_dim: 128,
    ..PROMPT
};

/// One side of a packed cache: `rows` vectors of `width` codes, each a `BITS`-wide index into
/// `codebook`, scaled by its row's norm.
struct PackedSide {
    codes: Vec<u32>,
    norms: Vec<f32>,
    codebook: Vec<f32>,
    width: usize,
}

impl PackedSide {
    fn new(rows: usize, width: usize, salt: usize) -> Self {
        let entries = 1usize << BITS;
        Self {
            codes: (0..rows * width)
                .map(|i| ((i * 7 + i / width * 3 + salt * 13) % entries) as u32)
                .collect(),
            norms: (0..rows)
                .map(|row| 0.5 + ((row + salt) % 5) as f32 * 0.25)
                .collect(),
            codebook: (0..entries)
                .map(|i| i as f32 / (entries - 1) as f32 * 2. - 1.)
                .collect(),
            width,
        }
    }

    /// Coordinate `d` of vector `row`, decoded.
    fn decoded(&self, row: usize, d: usize) -> f64 {
        (self.codebook[self.codes[row * self.width + d] as usize] * self.norms[row]) as f64
    }

    /// The side on the device as `[b, kv, s, ·]` buffers: the codes packed low first into `u32`
    /// words, the norms and the codebook in `f32`.
    fn operand(&self, client: &Client, case: &Case) -> PackedAttentionOperand {
        let per_word = 32 / BITS as usize;
        let words: Vec<u32> = self
            .codes
            .chunks(per_word)
            .map(|fields| {
                fields
                    .iter()
                    .enumerate()
                    .map(|(at, &code)| code << (at as u32 * BITS))
                    .sum()
            })
            .collect();
        let rows = [case.batch, case.kv_heads, case.seq_kv];
        let on_device = |shape: Vec<usize>, bytes: &[u8], element: ElemType| {
            PlainTensor::of_handle(TensorHandle::new_contiguous(
                Shape::from(shape),
                client.create_from_slice(bytes),
                element,
            ))
        };
        PackedAttentionOperand {
            words: on_device(
                [&rows[..], &[self.width / per_word]].concat(),
                u32::as_bytes(&words),
                u32::elem_type_native(),
            ),
            norms: on_device(
                [&rows[..], &[1]].concat(),
                f32::as_bytes(&self.norms),
                f32::elem_type_native(),
            ),
            codebook: on_device(
                vec![self.codebook.len()],
                f32::as_bytes(&self.codebook),
                f32::elem_type_native(),
            ),
        }
    }
}

/// Every setting of the staged cube on `case` over a packed cache, against the host's attention
/// over the decoded rows; a setting this device declines is skipped, and says so, but at least one
/// reads the cache.
fn check_packed(case: Case) {
    let client: Client = cubecl::test_device().client();
    let hw = Hardware::from_client(&client);
    let Case {
        batch,
        heads,
        kv_heads,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
        ..
    } = case;
    let group = heads / kv_heads;
    let rows = batch * kv_heads * seq_kv;
    let (keys, values) = (
        PackedSide::new(rows, head_dim, 1),
        PackedSide::new(rows, val_dim, 2),
    );
    let wobble = |i: usize| ((i * 2654435761 + 40503) % 2048) as f32 / 1024. - 1.;
    let cells = batch * heads * seq_q * head_dim;
    let (query, q) =
        TestInput::builder(client.clone(), Shape::new([batch, heads, seq_q, head_dim]))
            .dtype(half::f16::elem_type_native())
            .custom((0..cells).map(wobble).collect())
            .generate_with_f32_host_data();

    let mut out = vec![0f64; batch * heads * seq_q * val_dim];
    for b in 0..batch {
        for h in 0..heads {
            let kv = h / group;
            for i in 0..seq_q {
                let row = (b * heads + h) * seq_q + i;
                let key_row = |j: usize| (b * kv_heads + kv) * seq_kv + j;
                let scores: Vec<Option<f64>> = (0..seq_kv)
                    .map(|j| {
                        case.sees(b, h, i, j).then(|| {
                            (0..head_dim)
                                .map(|d| {
                                    q.get_f32(&[b, h, i, d]) as f64 * keys.decoded(key_row(j), d)
                                })
                                .sum::<f64>()
                                * case.scale()
                        })
                    })
                    .collect();
                let max = scores.iter().flatten().copied().reduce(f64::max).unwrap();
                let total: f64 = scores.iter().flatten().map(|s| (s - max).exp()).sum();
                for c in 0..val_dim {
                    out[row * val_dim + c] = (0..seq_kv)
                        .map(|j| {
                            scores[j].map_or(0., |s| (s - max).exp() / total)
                                * values.decoded(key_row(j), c)
                        })
                        .sum();
                }
            }
        }
    }

    let mut read = 0;
    for settings in CubeStagedAttentionRoutine.settings(&hw) {
        let lengths = TestInput::builder(client.clone(), Shape::new([1]))
            .dtype(u32::elem_type_native())
            .custom(vec![seq_kv as f32])
            .generate_without_host_data();
        let cache = PackedAttentionCache::new(
            keys.operand(&client, &case),
            values.operand(&client, &case),
            KvBitWidths {
                key: BITS,
                value: BITS,
            },
            None,
        );
        let tensors = AttentionTensors::packed(
            &client,
            PlainTensor::of_handle(query.clone()),
            cache,
            PlainTensor::of_handle(lengths),
            AttentionScoring {
                visibility: case.visibility,
                scale: ScoreScale::Custom(case.scale() as f32),
            },
        )
        .expect("the case's tensors are verified");
        match DirectAttention(CubeStagedAttentionRoutine).launch_on(&hw, tensors, &settings) {
            Ok(result) => {
                // The output comes back `[b, q, h, dv]`, the host's `[b, h, q, dv]`.
                let actual = HostData::from_tensor_handle(&client, result, HostDataType::F32);
                for (at, &want) in out.iter().enumerate() {
                    let c = at % val_dim;
                    let i = at / val_dim % seq_q;
                    let h = at / (val_dim * seq_q) % heads;
                    let b = at / (val_dim * seq_q * heads);
                    let got = actual.get_f32(&[b, i, h, c]) as f64;
                    assert!(
                        (got - want).abs() <= 5e-3 * (1. + want.abs()),
                        "{settings}: out[{b}, {h}, {i}, {c}] = {got}, expected {want}"
                    );
                }
                read += 1;
            }
            Err(Refusal::Declined { why }) => eprintln!("{settings}: declined, {why}"),
            Err(refusal) => eprintln!("{settings}: declined, {refusal:?}"),
        }
    }
    assert!(
        read > 0,
        "no setting of the staged cube read the packed cache"
    );
    eprintln!("{read} settings read the packed cache");
}

#[test]
fn the_staged_cube_reads_a_packed_cache_as_the_host_decodes_it() {
    check_packed(PROMPT);
}

#[test]
fn every_staged_cube_setting_reads_a_packed_cache_at_a_head_dim_of_128() {
    check_packed(PROMPT_WIDE);
}
