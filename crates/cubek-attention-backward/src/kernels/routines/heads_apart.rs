//! How a walk owned by the keys places its cubes with the heads of a group apart
//! ([`KeyHeads::Apart`](crate::kernels::KeyHeads::Apart)).

use cubecl::tile::Levels;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::{G, S};

/// What heads apart decline: a group of one head has none to set apart.
pub(crate) const ONE_HEAD_A_GROUP: Refusal = Refusal::Declined {
    why: "the key walk's heads apart need a group of more than one head",
};

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
