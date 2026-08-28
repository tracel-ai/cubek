//! The extents and strides a tile is addressed by, in the kernel.

use cubecl::prelude::*;

use crate::*;

/// One operand's physical extents and strides, in scalars, one entry per physical axis.
///
/// The two are one value because they are never separately true: the tile constructor reads them
/// in step, counting the innermost extent in lines and dividing every coarser stride by the
/// served width. Held apart they are two `Coords<u32>` a caller can state at two ranks — the
/// short one panics on an opaque `Sequence` index and the long one silently ignores its tail —
/// and two same-typed arguments a call site can swap in silence. [`push`](Self::push) takes a
/// dim's extent and stride together, so neither is expressible.
///
/// A bound operand reads its geometry off the tensor ([`of_tensor`](Self::of_tensor)); one with
/// no address states it, which is what [`Tile::of_sink`] and [`Tile::of_source`] take.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct RuntimeGeometry {
    pub(crate) shape: Coords<u32>,
    pub(crate) strides: Coords<u32>,
}

#[cube]
impl RuntimeGeometry {
    /// An empty geometry, grown a dim at a time by `push`.
    // A cube constructor, so there is no `Default` to implement: `default()` has no expansion
    // and nothing inside a kernel could call it. Same as `Coords::new`.
    #[allow(clippy::new_without_default)]
    pub fn new() -> RuntimeGeometry {
        RuntimeGeometry {
            shape: Coords::<u32>::new(),
            strides: Coords::<u32>::new(),
        }
    }

    /// The geometry a launched tensor carries, over its first `rank` dims — the rank the
    /// operand's projection addresses, which is not always the tensor's own.
    pub fn of_tensor<E: CubePrimitive>(
        tensor: &Tensor<E>,
        #[comptime] rank: usize,
    ) -> RuntimeGeometry {
        let mut geometry = RuntimeGeometry::new();
        #[unroll]
        for i in 0..rank {
            geometry.push(tensor.shape(i) as u32, tensor.stride(i) as u32);
        }
        geometry
    }

    /// One physical dim, coarsest first: how far it runs, and the scalar stride that steps it.
    pub fn push(&mut self, extent: u32, stride: u32) {
        self.shape.push(extent);
        self.strides.push(stride);
    }
}
