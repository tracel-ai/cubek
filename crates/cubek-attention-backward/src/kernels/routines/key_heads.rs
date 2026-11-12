//! How a walk owned by the keys takes the heads of a group, which all read the same keys:
//! [`KeyHeads`].

use cubecl::tile::Levels;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::{G, S};

/// What heads apart decline: a group of one head has none to set apart.
pub(crate) const ONE_HEAD_A_GROUP: Refusal = Refusal::Declined {
    why: "the key walk's heads apart need a group of more than one head",
};

/// How the key walk's cubes take the heads of a group, which all read the same keys: which
/// routines a strategy launches for its keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyHeads {
    /// Each cube walks every head of the group in turn and writes its keys' gradients.
    Together,
    /// Each cube walks one head and adds its part of the keys' gradients into sums every cube of
    /// the group adds into, which a second routine casts once they are done. Under a causal mask
    /// the cube with the first keys walks every query of every head; apart, its heads run side by
    /// side.
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

/// The cube level of a walk owned by the keys over `tiling` with the heads of a group of `group`
/// apart: the cubes along the keys and along the heads.
///
/// # Errors
///
/// [`ONE_HEAD_A_GROUP`] for a group of one head.
pub(crate) fn cubes_with_heads_apart(tiling: Levels, group: usize) -> Result<Levels, Refusal> {
    match group > 1 {
        true => Ok(tiling.cubes(&[S, G]).across(G, group)),
        false => Err(ONE_HEAD_A_GROUP),
    }
}
