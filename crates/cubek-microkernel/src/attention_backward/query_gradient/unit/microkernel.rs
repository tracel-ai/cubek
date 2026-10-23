use super::base::UnitQueryGradientMicrokernel;
use crate::attention::{D, S};
use crate::matmul::dense::unit::unit_registers_microkernel;
use cubecl::tile::{Region, Semiring, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One unit's queries of one head: `delta = Σ_dv out_gradient · out`, then
/// `query_gradient = scale · Σ_s probabilities ⊙ (out_gradient · valueᵀ − delta) · key` over the
/// first `steps` steps of the queries' walk through `S`, summed in `EA`, the probabilities read
/// back from each query's `logsumexp`.
///
/// `queries` is the region the unit owns, its walk the steps of keys; every tile is already
/// windowed to it. The bias is where the masking lies, read at each score's own coordinates. The
/// dot product is written before the walk, for the keys' walk to read.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn unit_query_gradient_microkernel<EA: Float>(
    queries: &Region,
    query_gradient: &Tile,
    delta: &Tile,
    query: &Tile,
    key: &Tile,
    value: &Tile,
    out: &Tile,
    out_gradient: &Tile,
    logsumexp: &Tile,
    bias: &Tile,
    steps: usize,
    scale: EA,
    #[comptime] microkernel: UnitQueryGradientMicrokernel,
) {
    // A tile's clone is a handle to the same cells: the dot product lands in `delta`.
    let mut delta = delta.clone();
    delta.dot::<EA>(out_gradient, out);
    let mut sum = query_gradient.accumulator::<EA>(
        bias,
        key,
        comptime!(microkernel.query_gradient.tile_instruction()),
        Semiring::SUM_PROD,
    );
    sum.zero();
    for step in queries.walk().range(0, steps) {
        let (query, key, value) = (query.at(&step), key.at(&step), value.at(&step));
        let (out_gradient, bias) = (out_gradient.at(&step), bias.at(&step));
        let mut score = bias.accumulator::<EA>(
            &query,
            &key,
            comptime!(microkernel.score.tile_instruction()),
            Semiring::SUM_PROD,
        );
        score.zero();
        for depth in step.walk() {
            unit_registers_microkernel(&depth, &score, &query, &key);
        }
        score.along(S).probabilities::<EA>(scale, &bias, logsumexp);
        let mut gradient = bias.accumulator::<EA>(
            &out_gradient,
            &value,
            comptime!(microkernel.score_gradient.tile_instruction()),
            Semiring::SUM_PROD,
        );
        gradient.zero();
        for depth in step.walk() {
            unit_registers_microkernel(&depth, &gradient, &out_gradient, &value);
        }
        gradient
            .along(S)
            .softmax_gradient::<EA>(&score.along(S), &delta);
        let step_sum = sum.at(&step);
        for depth in step.walk() {
            unit_registers_microkernel(&depth, &step_sum, &gradient, &key);
        }
    }
    let mut scales = Array::<EA>::new(comptime!(microkernel.seq_q()));
    #[unroll]
    for q in 0..comptime!(microkernel.seq_q()) {
        scales[q] = scale;
    }
    sum.along(D).mul(&scales);
    sum.drained_into(query_gradient);
}
