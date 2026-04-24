//! Matmul-specific interpretation of the generic [CubeMapping] from `cubek-std`.
//!
//! The partitioned matmul interprets the `(x, y, z)` problem-space axes
//! as `(m, n, batch)`. GEMV variants use [cube_pos_to_matrix_batch] instead.

use cubecl::prelude::*;

pub use cubek_std::cube_count::{CubeMapping, CubeMappingLaunch, cube_mapping_launch};

#[cube]
/// Reads the cube position as matmul tensor coordinates `(m, n, batch)`.
pub fn cube_pos_to_m_n_batch(cube_mapping: &CubeMapping) -> (u32, u32, u32) {
    cube_mapping.cube_pos_to_xyz()
}

#[cube]
/// Reads the cube position as GEMV `(matrix_axis, batch)` coordinates.
///
/// GEMV is 2D in problem space (the matrix axis + batch). The routine builds
/// its [CubeCountPlan] with `y = 1`, so the meaningful matrix-axis lives in `x`.
pub fn cube_pos_to_matrix_batch(cube_mapping: &CubeMapping) -> (u32, u32) {
    let (matrix, _, batch) = cube_mapping.cube_pos_to_xyz();
    (matrix, batch)
}
