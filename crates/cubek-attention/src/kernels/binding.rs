use crate::kernels::PackedCacheBinding;
use cubecl::prelude::TensorArg;
use cubecl::tile::launch::Bound;
use cubecl::tile::{Launcher, TileTensorLaunch};

/// The arguments of one launch, its operands bound under its launcher. Not `Debug`: the tile DSL's
/// `Launcher` is not.
pub struct AttentionBinding {
    pub launcher: Launcher,
    pub query: Bound,
    pub cache: AttentionCacheBinding,
    pub out: Bound,
    /// Each query's logsumexp, where the tensors ask for it.
    pub logsumexp: Option<Bound>,
    /// Which keys each query does not see, over `[B, KV, G, Q, S]`, where the tensors carry a
    /// mask.
    pub mask: Option<Bound>,
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
        key: TileTensorLaunch,
        value: TileTensorLaunch,
    },
    /// As a packed cache's tiles, which the kernel decodes as it reads them.
    Packed(PackedCacheBinding),
}

impl AttentionCacheBinding {
    /// The keys' and the values' tiles, as a stage that decodes what it is filled from reads
    /// them: a packed cache's are its words, read through its codebooks under its norms.
    ///
    /// # Panics
    ///
    /// On a packed cache whose keys carry the QJL fix-up, a second tile no single stage of keys
    /// holds: a routine that reads tiles declines it in its `prepare`.
    pub fn into_tiles(self) -> (TileTensorLaunch, TileTensorLaunch) {
        match self {
            Self::Tiles { key, value } => (key, value),
            Self::Packed(PackedCacheBinding {
                key,
                value,
                fixup: None,
            }) => (key, value),
            Self::Packed(_) => panic!("a stage of keys holds one tile, and a fix-up is a second"),
        }
    }
}
