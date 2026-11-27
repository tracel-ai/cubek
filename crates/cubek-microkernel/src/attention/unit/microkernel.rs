use super::base::UnitAttentionMicrokernel;
use crate::attention::{DV, S};
use crate::matmul::unit::unit_registers_microkernel;
use cubecl::tile::ops::softmax::OnlineSoftmax;
use cubecl::tile::{Region, Semiring, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One unit's rows of one head: `out = softmax(query · keyᵀ · scale + bias) · value`, summed in
/// `EA` over the first `steps` steps of the rows' walk through `S`.
///
/// `rows` is the region the unit owns, its walk the steps of keys; `out`, `logsumexp`, `query`,
/// `key`, `value` and `bias` are already windowed to it. The bias is where the masking lies, read
/// at each score's own coordinates. Each step is taken into the rows by `unit_attention_step`,
/// then the rows are normalized and drained into `out`, and each row's logsumexp written where the
/// caller asks for it.
///
/// The scale each side shares whole is factored out of it ([`Tile::factor`]) and applied once: the
/// keys' to every score the softmax reads, the values' to the output as it drains. The probabilities are contracted with the values from the score's
/// registers, which read the values as they lie.
///
/// Where the scores take sinks, `sinks` is the bias they are taken in under
/// ([`VisibleKeys::sink_bias`](crate::attention::VisibleKeys::sink_bias)), windowed to `rows`: the
/// unit holds every key of its rows, so it opens their softmax at their sinks before its walk.
#[cube]
pub fn unit_attention_microkernel<EA: Float>(
    rows: &Region,
    out: &Tile,
    logsumexp: &ComptimeOption<Tile>,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &Tile,
    sinks: &ComptimeOption<Tile>,
    steps: usize,
    scale: EA,
    #[comptime] microkernel: UnitAttentionMicrokernel,
) {
    let value_matmul = microkernel.value.software_matmul();
    let (value, value_scale) = value.factor();
    let mut sum = out.register_accumulator::<EA>(bias, &value, value_matmul, Semiring::SumProd);
    sum.zero();

    let mut softmax = OnlineSoftmax::<EA>::along(bias, S);

    #[comptime]
    match sinks {
        ComptimeOption::Some(sinks) => {
            let first = rows.walk().region(0);
            unit_attention_sinks::<EA>(
                &mut softmax,
                &query.at(&first),
                &key.at(&first),
                &bias.at(&first),
                &sinks.at(&first),
                microkernel,
            );
        }
        ComptimeOption::None => {}
    }

    for step in rows.walk().range(0, steps) {
        unit_attention_step::<EA>(
            &step,
            &mut sum.at(&step),
            &mut softmax,
            &query.at(&step),
            &key.at(&step),
            &value.at(&step),
            &bias.at(&step),
            scale,
            microkernel,
        );
    }

    sum.along(DV).mul(&softmax.recip_l());
    sum.mul(&value_scale).drained_into(out);

    #[comptime]
    match logsumexp {
        ComptimeOption::Some(logsumexp) => softmax.store_logsumexp(bias, &mut logsumexp.clone()),
        ComptimeOption::None => {}
    }
}

/// A unit's rows' sinks taken into their softmax before any key: a zero score over the walk's
/// first step, opened as the step's is, taken in under `sinks`, which is each row's sink at the
/// first key and masked elsewhere, so each row's max becomes its sink and its sum one. Its
/// probabilities weigh no value. `query`, `key`, `bias` and `sinks` are windowed to the step.
#[cube]
fn unit_attention_sinks<EA: Float>(
    softmax: &mut OnlineSoftmax<EA>,
    query: &Tile,
    key: &Tile,
    bias: &Tile,
    sinks: &Tile,
    #[comptime] microkernel: UnitAttentionMicrokernel,
) {
    let score_matmul = microkernel.score.software_matmul();
    let mut score = bias.register_accumulator::<EA>(query, key, score_matmul, Semiring::SumProd);
    score.zero();
    let _opened = softmax.step(&score, sinks, EA::from_int(1));
}

/// One step of keys taken into a unit's rows: the step's score opened over it and contracted, the
/// softmax taking it in, and the value adding the probabilities against the step's values into
/// the rows' `sum`, both multiplications by the shared unit microkernel.
///
/// `step` is the unit's region of the step, its walk the score's walk through `D`, one step of the
/// whole head dim, which holds the step's keys whole for the value too; `sum`, `query`, `key`,
/// `value` and `bias` are windowed to it, wherever the keys and values lie. The keys' per-tensor
/// scale is taken out of them, and the softmax reads the score under it.
#[cube]
pub(crate) fn unit_attention_step<EA: Float>(
    step: &Region,
    sum: &mut Tile,
    softmax: &mut OnlineSoftmax<EA>,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    bias: &Tile,
    scale: EA,
    #[comptime] microkernel: UnitAttentionMicrokernel,
) {
    let score_matmul = microkernel.score.software_matmul();
    let mut score = bias.register_accumulator::<EA>(query, key, score_matmul, Semiring::SumProd);
    score.zero();
    let (key, key_scale) = key.factor();

    for depth in step.walk() {
        unit_registers_microkernel(&depth, &score, query, &key);
    }

    let correction = softmax.step(&score.mul(&key_scale), bias, scale);
    sum.along(DV).rescale(&correction);

    // A tile's clone is a handle to the same cells: the shared microkernel adds into the sum
    // through it.
    for depth in step.walk() {
        unit_registers_microkernel(&depth, &sum.clone(), &score, value);
    }
}
