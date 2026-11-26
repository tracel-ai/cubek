use crate::kernels::PackedCacheBinding;
use cubecl::prelude::TensorArg;
use cubecl::tile::launch::Decline;
use cubecl::tile::{Launcher, TileTensorLaunch};
use cubek_std::routine::Breakable;

/// What a routine that stages its keys rests on: a packed cache whose keys carry the QJL fix-up
/// reaches no such routine.
const A_STAGE_IS_ONE_TILE: &str = "a stage of keys should be one tile, and a fix-up is a second";

/// A QJL fix-up reaching a routine that stages its keys.
const FIXUP_IN_A_STAGE: Decline = Decline::Broken {
    condition: A_STAGE_IS_ONE_TILE,
    cause: None,
};

/// The arguments of one launch, its operands bound under its launcher. Not `Debug`: the tile DSL's
/// `Launcher` is not.
pub struct AttentionBinding {
    pub launcher: Launcher,
    /// The query, `[b, kv, g, q, d]`.
    pub query: TileTensorLaunch,
    /// The keys and the values.
    pub cache: AttentionCacheBinding,
    /// The output, `[b, kv, g, q, dv]`.
    pub out: TileTensorLaunch,
    /// Each query's logsumexp, `[b, kv, g, q]`, where the tensors ask for it.
    pub logsumexp: Option<TileTensorLaunch>,
    /// Which keys each query does not see, over `[B, KV, G, Q, S]`, where the tensors carry a
    /// mask.
    pub mask: Option<TileTensorLaunch>,
    /// The attended length of each sequence.
    pub lengths: TensorArg,
    /// Where every sequence's queries lie end to end, their cumulative counts, one a sequence.
    pub query_ends: Option<TensorArg>,
    /// The queries a sequence has, or, where they lie end to end, all of them: what caps their
    /// ends.
    pub query_count: u32,
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
    /// # Errors
    ///
    /// [`Decline::Broken`] for a packed cache whose keys carry the QJL fix-up, a second tile no
    /// single stage of keys holds: a routine that reads tiles declines it in its `prepare`.
    pub fn into_tiles(self) -> Result<(TileTensorLaunch, TileTensorLaunch), Decline> {
        match self {
            Self::Tiles { key, value } => Ok((key, value)),
            Self::Packed(PackedCacheBinding {
                key,
                value,
                fixup: None,
            }) => Ok((key, value)),
            Self::Packed(_) => Err(FIXUP_IN_A_STAGE).or_broken(A_STAGE_IS_ONE_TILE),
        }
    }
}
