//! Online logsumexp 2-tuple (m, l) register update step.

use cubecl::prelude::*;

/// Pure register state transition step for online logsumexp:
/// Given current running max `m_curr`, current sum `l_curr`, and new `score`:
/// returns `(m_new, l_new, correction, weight)` where:
/// - `m_new`: updated running maximum
/// - `l_new`: updated running sum of exponentials
/// - `correction`: rescale factor `exp(m_curr - m_new)` for existing accumulator
/// - `weight`: new score weight `exp(score - m_new)` for incoming value
#[cube]
pub fn step<E: Float>(m_curr: E, l_curr: E, score: E) -> (E, E, E, E) {
    let m_new = max(m_curr, score);
    let correction = (m_curr - m_new).exp();
    let weight = (score - m_new).exp();
    let l_new = l_curr * correction + weight;
    (m_new, l_new, correction, weight)
}
