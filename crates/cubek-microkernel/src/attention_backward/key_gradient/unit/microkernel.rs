use super::base::UnitKeyGradientMicrokernel;
use crate::attention::{D, Q};
use crate::matmul::dense::unit::unit_registers_microkernel;
use cubecl::tile::{Region, Semiring, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One unit's keys of one key head:
/// `value_gradient = Σ_{g, q} probabilities · out_gradient` and
/// `key_gradient = scale · Σ_{g, q} probabilities ⊙ (value · out_gradientᵀ − delta) · query`, the
/// scores held keys first, summed in `EA`, the probabilities read back from each query's
/// `logsumexp`. Each head of the group walks its steps of queries from `first_step` on: the
/// queries before it see none of the unit's keys.
///
/// `keys` is the region the unit owns, its walk the heads of the group and theirs the steps of
/// queries; every tile is already windowed to it. The bias is where the masking lies, read at each
/// score's own coordinates.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn unit_key_gradient_microkernel<EA: Float>(
    keys: &Region,
    key_gradient: &Tile,
    value_gradient: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    out_gradient: &Tile,
    logsumexp: &Tile,
    delta: &Tile,
    bias: &Tile,
    first_step: usize,
    scale: EA,
    #[comptime] microkernel: UnitKeyGradientMicrokernel,
) {
    let mut key_sum = key_gradient.accumulator::<EA>(
        bias,
        query,
        comptime!(microkernel.key_gradient.tile_instruction()),
        Semiring::SUM_PROD,
    );
    key_sum.zero();
    let mut value_sum = value_gradient.accumulator::<EA>(
        bias,
        out_gradient,
        comptime!(microkernel.value_gradient.tile_instruction()),
        Semiring::SUM_PROD,
    );
    value_sum.zero();
    for head in keys.walk() {
        let queries = head.walk();
        let steps = queries.total();
        for step in queries.range(first_step, steps) {
            let (query, key, value) = (query.at(&step), key.at(&step), value.at(&step));
            let (out_gradient, bias) = (out_gradient.at(&step), bias.at(&step));
            let (logsumexp, delta) = (logsumexp.at(&step), delta.at(&step));
            let mut score = bias.accumulator::<EA>(
                &key,
                &query,
                comptime!(microkernel.score.tile_instruction()),
                Semiring::SUM_PROD,
            );
            score.zero();
            for depth in step.walk() {
                unit_registers_microkernel(&depth, &score, &key, &query);
            }
            score.along(Q).probabilities::<EA>(scale, &bias, &logsumexp);
            let step_value_sum = value_sum.at(&step);
            for depth in step.walk() {
                unit_registers_microkernel(&depth, &step_value_sum, &score, &out_gradient);
            }
            let mut gradient = bias.accumulator::<EA>(
                &value,
                &out_gradient,
                comptime!(microkernel.score_gradient.tile_instruction()),
                Semiring::SUM_PROD,
            );
            gradient.zero();
            for depth in step.walk() {
                unit_registers_microkernel(&depth, &gradient, &value, &out_gradient);
            }
            gradient
                .along(Q)
                .softmax_gradient::<EA>(&score.along(Q), &delta);
            let step_key_sum = key_sum.at(&step);
            for depth in step.walk() {
                unit_registers_microkernel(&depth, &step_key_sum, &gradient, &query);
            }
        }
    }
    let mut scales = Array::<EA>::new(comptime!(microkernel.seq_kv()));
    #[unroll]
    for s in 0..comptime!(microkernel.seq_kv()) {
        scales[s] = scale;
    }
    key_sum.along(D).mul(&scales);
    key_sum.drained_into(key_gradient);
    value_sum.drained_into(value_gradient);
}
