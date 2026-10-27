use crate::kernels::PackedCacheBinding;
use cubecl::prelude::TensorArg;
use cubecl::tile::launch::Bound;
use cubecl::tile::{Launcher, TileArgLaunch};

/// The arguments of one launch, its operands bound under its launcher. Not `Debug`: the tile DSL's
/// `Launcher` is not.
pub struct AttentionBinding {
    pub launcher: Launcher,
    pub query: Bound,
    pub cache: AttentionCacheBinding,
    pub out: Bound,
    /// Each query's logsumexp, where the tensors ask for it.
    pub logsumexp: Option<Bound>,
    /// The attended length of each sequence.
    pub lengths: TensorArg,
}

/// How a launch's keys and values are bound.
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, consumed by its dispatch"
)]
pub enum AttentionCacheBinding {
    /// As tiles, under the launcher.
    Tiles {
        key: TileArgLaunch,
        value: TileArgLaunch,
    },
    /// As a packed cache's tiles, which the kernel decodes as it reads them.
    Packed(PackedCacheBinding),
}

impl AttentionCacheBinding {
    /// The keys' and the values' tiles.
    ///
    /// # Panics
    ///
    /// On a packed cache, which a routine reading tiles declines in its `prepare`.
    pub fn into_tiles(self) -> (TileArgLaunch, TileArgLaunch) {
        match self {
            Self::Tiles { key, value } => (key, value),
            Self::Packed(_) => panic!("a packed cache binds no tile"),
        }
    }
}
