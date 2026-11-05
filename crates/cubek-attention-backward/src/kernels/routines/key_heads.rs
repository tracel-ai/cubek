//! How a walk owned by the keys takes the heads of a group, which all read the same keys:
//! [`KeyHeads`].

use cubecl::tile::launch::Refusal;
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{Levels, Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::attention::{G, S};

/// What heads apart decline: a group of one head has none to set apart.
pub(crate) const ONE_HEAD_A_GROUP: Refusal = Refusal::Declined {
    why: "the key walk's heads apart need a group of more than one head",
};

/// How the key walk's cubes take the heads of a group, which all read the same keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyHeads {
    /// Each cube walks every head of the group in turn and writes its keys' gradients.
    Together,
    /// Each cube walks one head and adds its part of the keys' gradients into sums every cube of
    /// the group adds into, cast once they are done. Under a causal mask the cube with the first
    /// keys walks every query of every head; apart, its heads run side by side.
    Apart,
}

/// Nothing for heads together, so the settings that hold it keep their names.
impl core::fmt::Display for KeyHeads {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Together => Ok(()),
            Self::Apart => write!(f, " key_heads=apart"),
        }
    }
}

impl KeyHeads {
    /// The cube level of a walk owned by the keys over `tiling`: the cubes along the keys, and
    /// along the heads of a group of `group` too where the heads are apart.
    ///
    /// # Errors
    ///
    /// [`ONE_HEAD_A_GROUP`] for heads apart in a group of one head.
    pub(crate) fn cubes(self, tiling: Levels, group: usize) -> Result<Levels, Refusal> {
        match self {
            Self::Together => Ok(tiling.cubes(&[S])),
            Self::Apart if group > 1 => Ok(tiling.cubes(&[S, G]).across(G, group)),
            Self::Apart => Err(ONE_HEAD_A_GROUP),
        }
    }

    /// Whether `partitioning`'s cubes hold fewer heads than the `group`, so the cubes of a group
    /// add their keys' gradients into sums.
    pub(crate) fn apart_in(partitioning: &Partitioning, group: usize) -> bool {
        partitioning.box_of(ComputeScope::Cube).extent(G) < group
    }
}

/// Every plane of the launch whose cube took the first head of its group casting its keys' and
/// values' gradients out of the sums every cube of the group added into: the other cubes of the
/// group cover the same keys.
#[cube(launch_unchecked)]
pub fn cast_key_gradient_kernel(
    key_sum: &TileTensor<'_>,
    value_sum: &TileTensor<'_>,
    key_gradient: &TileTensor<'_>,
    value_gradient: &TileTensor<'_>,
    partitioning: Partitioning,
) {
    let (key_sum, value_sum) = (key_sum.tile(&partitioning), value_sum.tile(&partitioning));
    let (d_k, d_v) = (
        key_gradient.tile(&partitioning),
        value_gradient.tile(&partitioning),
    );

    for plane in &partitioning {
        if plane.origin(G) == 0 {
            let mut d_k = d_k.at(&plane);
            d_k.copy_cast_from(&key_sum.at(&plane));
            let mut d_v = d_v.at(&plane);
            d_v.copy_cast_from(&value_sum.at(&plane));
        }
    }
}
