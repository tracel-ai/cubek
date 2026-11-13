//! How a strategy's walk owned by the keys takes the heads of a group, which all read the same
//! keys ([`KeyHeads`]), and the routines it launches for it ([`KeyHeadsBlueprint`]).

use crate::kernels::{AttentionBackwardWalk, CastKeyGradientBlueprint, CastKeyGradientRoutine};
use cubek_std::routine::Grid;

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

/// What a strategy launches for its keys' and values' gradients, as its [`KeyHeads`] says: the
/// walk owned by the keys alone with the heads together, or the walk adding into sums then the
/// cast out of them with the heads apart. `W` is the walk's blueprint.
#[derive(Debug, Clone)]
pub enum KeyHeadsBlueprint<W> {
    /// The walk, which writes the gradients.
    Together(W),
    /// The walk, which adds into the [`KeyGradientSums`](crate::kernels::KeyGradientSums), then
    /// the cast, under its planes.
    Apart {
        sums: W,
        /// Boxed: the variant holds one launch more than the other.
        cast: Box<CastKeyGradientBlueprint>,
    },
}

impl<M, SM> KeyHeadsBlueprint<AttentionBackwardWalk<M, SM>> {
    /// `sums`, the walk adding into the sums, then the cast under its planes.
    pub(crate) fn apart(sums: AttentionBackwardWalk<M, SM>) -> Self {
        Self::Apart {
            cast: Box::new(CastKeyGradientRoutine.prepare(&sums)),
            sums,
        }
    }

    /// The walk's grid: the widest launch of every strategy that has one owned by the keys.
    pub(crate) fn grid(&self) -> &Grid {
        match self {
            Self::Together(walk) | Self::Apart { sums: walk, .. } => &walk.grid,
        }
    }
}
