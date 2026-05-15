//! Stage-shaped variant payloads: `StageTile` and `PartitionTile`/`PipelinedTile`.

pub(crate) mod partition;
pub(crate) mod stage_tile;

pub use partition::*;
pub use stage_tile::*;
