use cubecl::prelude::{ComptimeOptionArgs, Tensor, TensorArg};
use cubecl::tile::Launcher;
use cubecl::tile::launch::Bound;
use cubek_microkernel::attention::packed_cache::PackedCacheArgLaunch;

/// The arguments of one launch, its operands bound under its launcher. Not `Debug`: cubek's
/// `Launcher` is not.
pub struct AttentionBinding {
    pub launcher: Launcher,
    pub query: Bound,
    pub cache: AttentionCacheBinding,
    pub out: Bound,
    /// The attended length of each sequence.
    pub lengths: TensorArg,
    /// The page table where the keys and values lie in pools of pages, which the kernel reads
    /// them through.
    pub pages: ComptimeOptionArgs<&'static Tensor<u32>>,
}

/// How a launch's keys and values are bound.
#[expect(
    clippy::large_enum_variant,
    reason = "one per launch, consumed by its dispatch"
)]
pub enum AttentionCacheBinding {
    /// As tiles, under the launcher.
    Tiles { key: Bound, value: Bound },
    /// As a packed cache, which the kernel decodes as it reads it.
    Packed(PackedCacheArgLaunch),
}

impl AttentionCacheBinding {
    /// The keys' and the values' tiles.
    ///
    /// # Panics
    ///
    /// On a packed cache, which a routine reading tiles declines in its `prepare`.
    pub fn into_tiles(self) -> (Bound, Bound) {
        match self {
            Self::Tiles { key, value } => (key, value),
            Self::Packed(_) => panic!("a packed cache binds no tile"),
        }
    }
}
