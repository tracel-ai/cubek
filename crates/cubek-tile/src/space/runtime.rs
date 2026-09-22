//! Which buffer dim witnesses an axis's runtime extent: the source of a
//! [`Dynamic`](crate::Extent) axis's size, read off an operand that carries the axis whole.

use crate::{Axis, Projection};

/// The one physical dim of a projection whose bound is `axis`'s own extent: it carries `axis`
/// alone, at coefficient `1`. Absent for a gather (the dim holds a receptive field several axes
/// reach over), for storage tiling (the extent is the product over the dims the axis is split
/// across) and for a broadcast axis (the buffer holds nothing that sizes it).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Witness {
    dim: usize,
}

impl Witness {
    pub(crate) fn new(projection: &Projection, axis: Axis) -> Option<Self> {
        if !projection.addresses(axis) {
            return None;
        }
        match projection.carriers(axis)[..] {
            [pa] if projection.physical_axis(pa).is_identity(axis) => Some(Witness { dim: pa }),
            _ => None,
        }
    }

    /// The physical dim whose bound is the extent.
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PhysicalAxisMap, Projection, StorageTiling};

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);

    /// The discrimination the operation space rests on: a bound is an axis's own extent only when
    /// one dim carries that axis alone. A gather's dim holds a receptive field its axes reach over,
    /// and storage tiling splits the extent across dims, so neither bound is it.
    #[test]
    fn a_witness_is_one_dim_carrying_the_axis_alone() {
        let direct = Projection::direct(&[A, B]);
        assert_eq!(Witness::new(&direct, A).map(|w| w.dim()), Some(0));
        assert_eq!(Witness::new(&direct, B).map(|w| w.dim()), Some(1));

        let gathered = Projection::new(&[A, B], &[PhysicalAxisMap::affine(&[(A, 1), (B, 1)])]);
        assert_eq!(Witness::new(&gathered, A), None);
        assert_eq!(Witness::new(&gathered, B), None);

        let tiled = Projection::tiled(&[A, B], StorageTiling::per_axis(&[1, 2]));
        assert_eq!(Witness::new(&tiled, A).map(|w| w.dim()), Some(0));
        assert_eq!(Witness::new(&tiled, B), None);
    }
}
