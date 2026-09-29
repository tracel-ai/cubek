//! The asynchronous line copy under a straight or packed fill ([`Delivery::AsyncPerUnit`]): each unit hands
//! its lines to the copy engine rather than loading and storing them, so they land in the stage
//! without passing through its registers, after the fill has returned.
//!
//! Which lines a unit takes is the straight fill's; only who carries each one differs. Nothing
//! here waits: the slot's barrier is handed the copies and flips once they land
//! ([`Slot::release_write`](crate::Slot)), or a blocking copy waits on one of its own
//! ([`Memory::load_from`]).

use cubecl::{
    prelude::barrier::{Barrier, copy_async, copy_async_checked},
    prelude::*,
    std::tensor::layout::CoordsDyn,
};

use crate::*;

/// The widths of one copy the engine takes, in bytes: `cp.async` moves 4, 8 or 16.
const COPY_BYTES: [usize; 3] = [4, 8, 16];

/// Copy destination line `i` asynchronously: the source line at the same coordinates, whole, or
/// zeros where the source masks it off. It has not landed when this returns.
#[cube]
pub(crate) fn copy_line_async<I2: Numeric, WP2: Size, SW: Size>(
    d: &mut [Vector<I2, WP2>],
    s: &Masked<'_, Vector<I2, SW>, CoordsDyn>,
    layout: &BufferLayout,
    i: usize,
) {
    let width = WP2::value();
    let bytes = comptime!(width * I2::size().comptime());
    comptime!(assert!(
        COPY_BYTES.contains(&bytes),
        "copy_line_async: the copy engine moves 4, 8 or 16 bytes at once, and this stage's lines are \
         {bytes}: serve the operand in lines one copy moves"
    ));
    let offset = layout.line_offset(i);
    let run = s.item_run(layout.line_coords(i));
    let dst = &mut d[offset..offset + 1];
    // One type on both sides: the transport already refused a stage served at another width.
    let dst = dst.downcast_mut::<Vector<I2, SW>>();
    if comptime!(s.check) {
        copy_async_checked(run, dst, comptime!(width as u32));
    } else {
        copy_async(run, dst, comptime!(width as u32));
    }
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// Fill `self` from `src` and wait for it: [`fill_from`](Memory::fill_from), plus the barrier
    /// an async copy needs before its bytes can be read. What a blocking
    /// [`copy_from`](Tile::copy_from) runs; a pipelined fill leaves the wait to its slot.
    ///
    /// The contract is `fill_from`'s either way: a `sync_cube` must still separate the fill from
    /// its readers.
    pub(crate) fn load_from(&mut self, src: &Memory<T>, #[comptime] space: Space) {
        if comptime!(src.access.delivery == Delivery::AsyncPerUnit) {
            let landed = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
            self.fill_from(src, space);
            landed.commit_copy_async();
            landed.arrive_and_wait();
        } else {
            self.fill_from(src, space);
        }
    }
}
