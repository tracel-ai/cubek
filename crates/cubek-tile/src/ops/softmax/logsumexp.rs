//! Online logsumexp 2-tuple (m, l) register update step.

use cubecl::prelude::*;

/// One online-logsumexp step: from running max `m_curr`, running sum `l_curr` and a new `score`,
/// returns `(m_new, l_new, correction, weight)`: the updated max and sum, the accumulator rescale
/// `exp(m_curr - m_new)`, and the incoming value's weight `exp(score - m_new)`.
#[cube]
pub fn step<E: Float>(m_curr: E, l_curr: E, score: E) -> (E, E, E, E) {
    let m_new = max(m_curr, score);
    let correction = (m_curr - m_new).exp();
    let weight = (score - m_new).exp();
    let l_new = l_curr * correction + weight;
    (m_new, l_new, correction, weight)
}
