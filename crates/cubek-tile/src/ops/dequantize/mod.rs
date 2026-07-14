pub mod schedule;

use cubecl::prelude::*;

use self::schedule::dequantize_direct;
use crate::*;

#[cube]
impl<O: Numeric> Tile<O> {
    /// Copy `input` into `self`. Both tiles serve `O`; a quantized input dequantizes on read
    /// ([`Tile::flat`]), so the body is a plain flat copy. `I` is the input's storage element,
    /// threaded from the kernel (unused on a plain input). Per-tensor native only.
    pub fn dequantize_from<I: Numeric>(&mut self, input: &Tile<O>) {
        match comptime!(self.space.partitioner()) {
            Partitioner::Final(_) => dequantize_leaf::<I, O>(input, self),
            Partitioner::Level(level) => match level.schedule() {
                Schedule::Direct => dequantize_direct::<I, O>(input, self),
                _ => {
                    unimplemented!(
                        "currently unsupported schedule: {:?}. only {:?} is supported",
                        level.schedule(),
                        Schedule::Direct
                    );
                }
            },
        }
    }

    pub fn dequantize_from_at<I: Numeric>(&mut self, input: &Tile<O>, region: &Region) {
        self.at(region).dequantize_from::<I>(&input.at(region));
    }
}

/// The leaf: a flat elementwise copy; the read side dequantizes transparently when the input's
/// store carries [`QuantInfo`].
#[cube]
pub(crate) fn dequantize_leaf<I: Numeric, O: Numeric>(input: &Tile<O>, output: &mut Tile<O>) {
    // The physical widths are storage detail; input and output share the logical shape, so they
    // scan at the same width.
    let size!(W) = output.vector_size();

    let values = input.flat::<I, W>();
    let mut out = output.flat_mut::<W>();

    for i in 0..out.shape() {
        out.write(i, values.read(i));
    }
}
