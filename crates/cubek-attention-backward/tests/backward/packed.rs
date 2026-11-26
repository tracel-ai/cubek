//! A packed cache read as it lies by the staged cube: each step of keys and values decoded into
//! its stage once, for every query of the cube. Its output is the host's attention over the
//! decoded rows.

use super::reference::Case;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::zspace::Shape;
use cubek_attention::kernels::{
    AttentionLaunch, AttentionScoring, AttentionTensors, CubeStagedAttentionRoutine,
    DirectAttention, KvBitWidths, PackedAttentionCache, PackedAttentionOperand, ScoreScale,
};
use cubek_matmul::kernels::tensor::PlainTensor;
use cubek_microkernel::attention::{AttentionVisibility, QueryPlacement};
use cubek_std::Hardware;
use cubek_std::routine::Routine;
use cubek_test_utils::{
    HostAttention, HostData, HostDataType, PackedVectors, TestInput, TestOutcome, ValidationResult,
};

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
/// `codebook`, packed low first into `u32` words, and scaled by its row's norm.
struct PackedSide {
    words: Vec<u32>,
    norms: Vec<f32>,
    codebook: Vec<f32>,
    width: usize,
}

impl PackedSide {
    fn new(rows: usize, width: usize, salt: usize) -> Self {
        let entries = 1usize << BITS;
        let per_word = 32 / BITS as usize;
        let codes: Vec<u32> = (0..rows * width)
            .map(|i| ((i * 7 + i / width * 3 + salt * 13) % entries) as u32)
            .collect();
        Self {
            words: codes
                .chunks(per_word)
                .map(|fields| {
                    fields
                        .iter()
                        .enumerate()
                        .map(|(at, &code)| code << (at as u32 * BITS))
                        .sum()
                })
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

    /// The side's vectors as the host decodes them.
    fn vectors(&self) -> PackedVectors<'_> {
        PackedVectors {
            words: &self.words,
            norms: &self.norms,
            codebook: &self.codebook,
            bits: BITS,
            coords: self.width,
        }
    }

    /// The side on the device as `[b, kv, s, ·]` buffers: the words, the norms and the codebook in
    /// `f32`.
    fn operand(&self, client: &Client, case: &Case) -> PackedAttentionOperand {
        let per_word = 32 / BITS as usize;
        let words = &self.words;
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
                u32::as_bytes(words),
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
/// over the decoded rows: a broken setting fails, and a device on which every setting declines
/// reports a skip.
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

    let (decoded_keys, decoded_values) = (keys.vectors(), values.vectors());
    let host = HostAttention {
        scale: case.scale(),
    };
    let mut out = Vec::with_capacity(batch * heads * seq_q * val_dim);
    for b in 0..batch {
        for h in 0..heads {
            let kv = h / group;
            let key_row = |j: usize| (b * kv_heads + kv) * seq_kv + j;
            for i in 0..seq_q {
                let query: Vec<f64> = (0..head_dim)
                    .map(|d| q.get_f32(&[b, h, i, d]) as f64)
                    .collect();
                let seen: Vec<usize> = (0..seq_kv).filter(|&j| case.sees(b, h, i, j)).collect();
                out.extend(host.attend(
                    &query,
                    &seen,
                    |j, d| decoded_keys.decoded(key_row(j), d),
                    |j, c| decoded_values.decoded(key_row(j), c),
                    val_dim,
                ));
            }
        }
    }

    let (mut read, mut declined) = (0, Vec::new());
    for settings in CubeStagedAttentionRoutine::settings(&hw) {
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
            Err(broken) if broken.is_broken() => panic!("{settings}: {broken}"),
            Err(decline) => declined.push(format!("{settings}: {decline}")),
        }
    }
    if read == 0 {
        let why = format!("every setting of the staged cube declines: {declined:#?}");
        TestOutcome::Validated(ValidationResult::Skipped(why)).enforce();
    }
}

#[test]
fn the_staged_cube_reads_a_packed_cache_as_the_host_decodes_it() {
    check_packed(PROMPT);
}

#[test]
fn every_staged_cube_setting_reads_a_packed_cache_at_a_head_dim_of_128() {
    check_packed(PROMPT_WIDE);
}
