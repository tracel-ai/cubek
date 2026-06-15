//! Attention-specific interpretation of the generic [CubeMapping] from `cubek-std`.
//!
//! Attention has 2D problem-space axes: `(seq_q_tile, batch_heads)` where
//! `batch_heads = batch * num_heads`. The [HypercubeBlueprint], [CubeCountPlan]
//! and [CubeMapping] types come directly from `cubek-std`; this module only
//! adds the attention-specific `(seq_q, batch_heads)` mapper.

use cubecl;
use cubecl::prelude::*;

pub use cubek_std::cube_count::{
    CubeCountPlan, CubeMapping, CubeMappingLaunch, HypercubeBlueprint, cube_mapping_launch,
};

#[cube]
/// Reads the cube position as attention `(seq_q_index, batch_heads_index)` coordinates.
///
/// The `batch_heads_index` spans `batch * num_heads`; the third axis is unused.
pub fn cube_pos_to_q_batch_heads(cube_mapping: &CubeMapping) -> (u32, u32) {
    let (seq_q, batch_heads, _) = cube_mapping.cube_pos_to_xyz();
    (seq_q, batch_heads)
}
