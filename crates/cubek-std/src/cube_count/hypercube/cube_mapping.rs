use cubecl::prelude::*;

use crate::cube_count::{CubeCountPlan, CubeCountPlanKind, GlobalOrder, swizzle};

#[derive(CubeType, CubeLaunch)]
/// Runtime-side counterpart of [CubeCountPlan]: given the cube position,
/// resolves the conceptual `(x, y, z)` coordinates in problem space.
///
/// Each operation is responsible for mapping the returned generic `(x, y, z)`
/// tuple to its own domain axes (e.g. matmul interprets them as `(m, n, batch)`,
/// gemv as `(matrix_axis, _, batch)`, attention as `(seq_q, batch_heads, _)`).
pub struct CubeMapping {
    strategy: CubeMappingStrategy,
    #[cube(comptime)]
    pub can_yield_extra_cubes: bool,
    #[cube(comptime)]
    global_order: GlobalOrder,
}

#[derive(CubeType, CubeLaunch)]
/// [CubeCountPlanKind] stripped of non-essential runtime information.
///
/// Given as runtime input to kernels.
#[allow(unused)] // Constructed via CubeMappingStrategyArgs only
pub(crate) enum CubeMappingStrategy {
    FromProblem,
    SmFirst {
        x_cubes: u32,
        y_cubes: u32,
        z_cubes: u32,
    },
    CubeFirst {
        x_cubes: u32,
        y_cubes: u32,
        z_cubes: u32,
    },
    Flattened {
        x_cubes: u32,
        y_cubes: u32,
    },
    Spread {
        x_cubes: u32,
        y_cubes: u32,
        z_cubes: u32,
    },
}

#[cube]
impl CubeMapping {
    /// Returns the number of valid cubes (problem-space volume).
    pub fn num_valid_cubes(&self) -> usize {
        match &self.strategy {
            CubeMappingStrategy::FromProblem | CubeMappingStrategy::Flattened { .. } => {
                panic!("Shouldn't need to be called because the cube count should always be exact")
            }
            CubeMappingStrategy::SmFirst {
                x_cubes,
                y_cubes,
                z_cubes,
            }
            | CubeMappingStrategy::CubeFirst {
                x_cubes,
                y_cubes,
                z_cubes,
            }
            | CubeMappingStrategy::Spread {
                x_cubes,
                y_cubes,
                z_cubes,
            } => *x_cubes as usize * *y_cubes as usize * *z_cubes as usize,
        }
    }

    /// Given a cube position, returns the generic problem-space coordinates `(x, y, z)`.
    ///
    /// Consumers assign meaning to `x/y/z` (matmul: `m/n/batch`, gemv: `matrix/_/batch`, etc.).
    pub fn cube_pos_to_xyz(&self) -> (u32, u32, u32) {
        match &self.strategy {
            CubeMappingStrategy::FromProblem => (CUBE_POS_X, CUBE_POS_Y, CUBE_POS_Z),

            CubeMappingStrategy::SmFirst {
                x_cubes, y_cubes, ..
            } => {
                self.strategy
                    .absolute_index_to_xyz(CUBE_POS, *x_cubes, *y_cubes, self.global_order)
            }

            CubeMappingStrategy::CubeFirst {
                x_cubes, y_cubes, ..
            } => self.strategy.absolute_index_to_xyz(
                CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize,
                *x_cubes,
                *y_cubes,
                self.global_order,
            ),

            CubeMappingStrategy::Flattened { x_cubes, y_cubes } => self
                .strategy
                .absolute_index_to_xyz(CUBE_POS_X as usize, *x_cubes, *y_cubes, self.global_order),

            CubeMappingStrategy::Spread {
                x_cubes, y_cubes, ..
            } => {
                self.strategy
                    .absolute_index_to_xyz(CUBE_POS, *x_cubes, *y_cubes, self.global_order)
            }
        }
    }
}

#[cube]
impl CubeMappingStrategy {
    fn absolute_index_to_xyz(
        &self,
        absolute_index: usize,
        x_cubes: u32,
        y_cubes: u32,
        #[comptime] global_order: GlobalOrder,
    ) -> (u32, u32, u32) {
        let z_stride = (x_cubes * y_cubes) as usize;
        let z_pos = absolute_index / z_stride;
        let xy_pos = absolute_index % z_stride;

        let (x_pos, y_pos) = match comptime!(global_order) {
            GlobalOrder::RowMajor => ((xy_pos / y_cubes as usize) as u32, xy_pos as u32 % y_cubes),
            GlobalOrder::ColMajor => (xy_pos as u32 % x_cubes, (xy_pos / x_cubes as usize) as u32),
            GlobalOrder::SwizzleRow(w) => {
                let (x, y) = swizzle(xy_pos, y_cubes as usize, w);
                (y, x)
            }
            GlobalOrder::SwizzleCol(w) => swizzle(xy_pos, x_cubes as usize, w),
        };

        (x_pos, y_pos, z_pos as u32)
    }
}

/// Build a [CubeMappingLaunch] from a resolved [CubeCountPlan].
pub fn cube_mapping_launch<R: Runtime>(cube_count_plan: &CubeCountPlan) -> CubeMappingLaunch<R> {
    CubeMappingLaunch::new(
        mapping_strategy(&cube_count_plan.kind),
        cube_count_plan.kind.can_yield_extra_cubes(),
        cube_count_plan.global_order,
    )
}

fn mapping_strategy<R: Runtime>(
    cube_count_plan_kind: &CubeCountPlanKind,
) -> CubeMappingStrategyArgs<R> {
    match cube_count_plan_kind {
        CubeCountPlanKind::FromProblem { .. } => CubeMappingStrategyArgs::FromProblem,

        CubeCountPlanKind::Sm {
            cubes_first,
            problem_count,
            ..
        } => {
            if *cubes_first {
                CubeMappingStrategyArgs::CubeFirst {
                    x_cubes: problem_count.x,
                    y_cubes: problem_count.y,
                    z_cubes: problem_count.z,
                }
            } else {
                CubeMappingStrategyArgs::SmFirst {
                    x_cubes: problem_count.x,
                    y_cubes: problem_count.y,
                    z_cubes: problem_count.z,
                }
            }
        }

        CubeCountPlanKind::Flattened { problem_count, .. } => CubeMappingStrategyArgs::Flattened {
            x_cubes: problem_count.x,
            y_cubes: problem_count.y,
        },

        CubeCountPlanKind::Spread { problem_count, .. } => CubeMappingStrategyArgs::Spread {
            x_cubes: problem_count.x,
            y_cubes: problem_count.y,
            z_cubes: problem_count.z,
        },
    }
}
