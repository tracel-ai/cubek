use crate::{definition::QRProblem, launch::QRStrategy};

/// Resolve [`QRStrategy::Auto`] into a concrete strategy for the given problem.
///
/// Benchmarks on CUDA (f32, square matrices, 2026-07) show the TSQR-inspired
/// blocked Householder routine is the fastest strategy from 128x128 upward —
/// 2-20x faster than every alternative at 256² and beyond — and within ~25%
/// of the best (Givens) below that, where every strategy completes in under
/// 2 ms anyway:
///
/// | Size  | CGR    | BAHT   | BahtTsqr | MGS    |
/// |-------|--------|--------|----------|--------|
/// | 32²   | 577µs  | 1.06ms | 777µs    | 726µs  |
/// | 64²   | 1.61ms | 3.51ms | 1.74ms   | 2.40ms |
/// | 128²  | 6.96ms | 13.0ms | 4.12ms   | 9.92ms |
/// | 256²  | 39.3ms | 32.8ms | 11.0ms   | 46.9ms |
/// | 512²  | 307ms  | 84.9ms | 29.6ms   | 293ms  |
/// | 1024² | 2.50s  | 253ms  | 108ms    | 2.08s  |
///
/// Givens' small edge below 64² is dispatch-overhead noise and not worth
/// trading away Householder's numerical stability, and sparsity does not
/// change the picture (every routine performs dense updates). So Auto always
/// selects [`QRStrategy::BahtTsqr`]. The `problem` parameter is kept so
/// future size-based crossovers (or an autotuner) can slot in without an
/// API change.
///
/// The returned strategy is never `Auto`.
pub fn select_strategy(problem: &QRProblem) -> QRStrategy {
    let _ = problem;
    QRStrategy::BahtTsqr
}
