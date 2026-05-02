//! Streaming progress tracker for CPU reference algorithms.
//!
//! Reference implementations are slow on bench-scale problems, so callers want
//! to display a progression bar while they run. The contract here is simple:
//! every reference declares a `total` (number of output writes) up-front and
//! bumps a counter once per output write. Callers read `current()` from
//! another thread to stream the progression.
//!
//! Granularity is per-output-write: for matmul that's one bump per output
//! cell, for reduce that's one bump per output position, for attention that's
//! one bump per (batch, head, seq_q) row, etc. References are free to declare
//! a coarser granularity if a per-cell bump would dominate runtime — the only
//! invariant is that `current()` reaches `total` by the time the reference
//! returns.

use std::sync::atomic::{AtomicU64, Ordering};

/// Counter shared between a reference algorithm (which bumps it) and a caller
/// (which polls it for streaming progression).
#[derive(Debug)]
pub struct Progress {
    total: AtomicU64,
    current: AtomicU64,
}

impl Progress {
    /// Empty progress with `total = 0`. The reference algorithm will set the
    /// real total via [`Self::set_total`] once it knows the problem shape.
    pub fn new() -> Self {
        Self {
            total: AtomicU64::new(0),
            current: AtomicU64::new(0),
        }
    }

    /// Pre-declared total: caller already knows the count and just wants to
    /// poll. References still call [`Self::set_total`] on entry, which is a
    /// no-op when the value matches.
    pub fn with_total(total: u64) -> Self {
        Self {
            total: AtomicU64::new(total),
            current: AtomicU64::new(0),
        }
    }

    /// Declare the total number of output writes. Called by the reference
    /// algorithm at entry, before the first [`Self::bump`].
    pub fn set_total(&self, total: u64) {
        self.total.store(total, Ordering::Relaxed);
    }

    /// Increment the counter by one output write.
    pub fn bump(&self) {
        self.current.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by `n` output writes — useful when a reference
    /// writes a contiguous run of outputs in one inner loop.
    pub fn bump_by(&self, n: u64) {
        self.current.fetch_add(n, Ordering::Relaxed);
    }

    pub fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    pub fn current(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// `current / total` clamped to `[0.0, 1.0]`. Returns `0.0` when total is
    /// zero (reference hasn't started or declared its total yet).
    pub fn fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.current() as f64 / total as f64).clamp(0.0, 1.0)
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self::new()
    }
}
