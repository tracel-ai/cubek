//! The [`Ring`]: a small ring of staged buffers — the loader a pipelined lowering
//! ping-pongs through. The buffers are just [`Tile`]s, so the ring is generic over
//! what backs them (shared memory, or register fragments elsewhere); the caller
//! builds them. [`stage`](Ring::stage) copies an operand sub-tile into a slot,
//! [`get`](Ring::get) reads it back. Double buffering is two buffers with the
//! driver rotating `slot = i % 2`.

use cubecl::prelude::*;

use super::*;

/// A ring of staged buffer [`Tile`]s — one per slot.
#[derive(CubeType)]
pub struct Ring<E: Numeric> {
    buffers: Sequence<Tile<E>>,
}

#[cube]
impl<E: Numeric> Ring<E> {
    /// Wrap a sequence of buffer tiles as a ring; its depth is their count.
    pub fn new(buffers: Sequence<Tile<E>>) -> Ring<E> {
        Ring::<E> { buffers }
    }

    /// Stage `src` into buffer `slot`.
    pub fn stage(&mut self, #[comptime] slot: usize, src: &Tile<E>) {
        self.buffers.index_mut(slot).stage(src);
    }

    /// The buffer at `slot` — the staged operand the leaf reads.
    pub fn get(&self, #[comptime] slot: usize) -> &Tile<E> {
        self.buffers.index(slot)
    }
}
