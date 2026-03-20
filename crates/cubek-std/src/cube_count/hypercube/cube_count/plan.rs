use cubecl::{CubeCount, Runtime};

use crate::cube_count::{
    CubeCountStrategy, CubeMappingLaunch, GlobalOrder, HypercubeBlueprint, SmAllocation,
    hypercube::cube_count::mapping::CubeMappingStrategyArgs,
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CubeCountPlan {
    pub global_order: GlobalOrder,
    pub kind: CubeCountPlanKind,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Count3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl From<(u32, u32, u32)> for Count3d {
    fn from(value: (u32, u32, u32)) -> Self {
        Count3d {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl Count3d {
    // Use u64 to avoid overflow when multiplying large cube counts.
    pub(crate) fn total(&self) -> u64 {
        self.x as u64 * self.y as u64 * self.z as u64
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CubeCountPlanKind {
    FromProblem {
        problem_count: Count3d,
    },
    Sm {
        cubes_first: bool,
        num_sms_used: u32,
        cubes_per_sm: u32,
        problem_count: Count3d,
        num_sms: u32,
        sm_usage: SmAllocation,
    },
    Flattened {
        problem_count: Count3d,
    },
    Spread {
        problem_count: Count3d,
        spread_count: Count3d,
    },
}

impl CubeCountPlan {
    // Will check if the wanted cube count plan is possible, otherwise will fallback to spread
    pub fn from_blueprint(
        blueprint: &HypercubeBlueprint,
        target_count: Count3d,
        max_cube_count: &(u32, u32, u32),
    ) -> CubeCountPlan {
        let (max_x, max_y, max_z) = *max_cube_count;

        let plan_kind = match blueprint.cube_count_strategy {
            CubeCountStrategy::FromProblem => {
                if target_count.x > max_x || target_count.y > max_y || target_count.z > max_z {
                    None
                } else {
                    Some(CubeCountPlanKind::FromProblem {
                        problem_count: target_count,
                    })
                }
            }
            CubeCountStrategy::Sm {
                cubes_first,
                num_sms,
                sm_usage,
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sm_usage.allocate(num_sms, target_count.total() as usize);

                if (cubes_per_sm >= if cubes_first { max_x } else { max_y })
                    || (num_sms_used >= if cubes_first { max_y } else { max_x })
                {
                    None
                } else {
                    Some(CubeCountPlanKind::Sm {
                        cubes_first,
                        num_sms_used,
                        cubes_per_sm,
                        problem_count: target_count,
                        num_sms,
                        sm_usage,
                    })
                }
            }
            CubeCountStrategy::Flattened => {
                if target_count.total() >= max_x as u64 {
                    None
                } else {
                    Some(CubeCountPlanKind::Flattened {
                        problem_count: target_count,
                    })
                }
            }
            CubeCountStrategy::Spread => None,
        };

        CubeCountPlan {
            global_order: blueprint.global_order,
            kind: plan_kind
                .unwrap_or_else(|| spread_cube_count_plan(target_count, max_x, max_y, max_z)),
        }
    }

    pub fn new_from_problem(target_count: Count3d) -> Self {
        Self {
            global_order: Default::default(),
            kind: CubeCountPlanKind::FromProblem {
                problem_count: target_count,
            },
        }
    }

    pub fn can_yield_extra_cubes(&self) -> bool {
        self.kind.can_yield_extra_cubes()
    }

    pub fn resolve(&self) -> CubeCount {
        self.kind.resolve()
    }

    pub fn as_args<R: Runtime>(&self) -> CubeMappingLaunch<R> {
        CubeMappingLaunch::new(
            self.kind.mapping_strategy(),
            self.kind.can_yield_extra_cubes(),
            self.global_order,
        )
    }
}

impl CubeCountPlanKind {
    fn can_yield_extra_cubes(&self) -> bool {
        match self {
            CubeCountPlanKind::FromProblem { .. } | CubeCountPlanKind::Flattened { .. } => false,

            CubeCountPlanKind::Sm {
                num_sms_used,
                cubes_per_sm,
                problem_count,
                ..
            } => (num_sms_used * cubes_per_sm) as u64 != problem_count.total(),

            CubeCountPlanKind::Spread {
                problem_count,
                spread_count,
            } => problem_count.total() != spread_count.total(),
        }
    }

    fn resolve(&self) -> CubeCount {
        match self {
            CubeCountPlanKind::FromProblem { problem_count } => {
                CubeCount::Static(problem_count.x, problem_count.y, problem_count.z)
            }

            CubeCountPlanKind::Sm {
                cubes_first,
                num_sms_used,
                cubes_per_sm,
                ..
            } => {
                if *cubes_first {
                    CubeCount::Static(*cubes_per_sm, *num_sms_used, 1)
                } else {
                    CubeCount::Static(*num_sms_used, *cubes_per_sm, 1)
                }
            }

            CubeCountPlanKind::Flattened { problem_count } => {
                CubeCount::Static(problem_count.total() as u32, 1, 1)
            }

            CubeCountPlanKind::Spread { spread_count, .. } => {
                CubeCount::Static(spread_count.x, spread_count.y, spread_count.z)
            }
        }
    }

    fn mapping_strategy<R: Runtime>(&self) -> CubeMappingStrategyArgs<R> {
        match self {
            CubeCountPlanKind::FromProblem { .. } => CubeMappingStrategyArgs::FromProblem,

            CubeCountPlanKind::Sm {
                cubes_first,
                problem_count,
                ..
            } => {
                if *cubes_first {
                    CubeMappingStrategyArgs::CubeFirst {
                        m_cubes: problem_count.x,
                        n_cubes: problem_count.y,
                        batch_cubes: problem_count.z,
                    }
                } else {
                    CubeMappingStrategyArgs::SmFirst {
                        m_cubes: problem_count.x,
                        n_cubes: problem_count.y,
                        batch_cubes: problem_count.z,
                    }
                }
            }

            CubeCountPlanKind::Flattened { problem_count, .. } => {
                CubeMappingStrategyArgs::Flattened {
                    m_cubes: problem_count.x,
                    n_cubes: problem_count.y,
                }
            }

            CubeCountPlanKind::Spread { problem_count, .. } => CubeMappingStrategyArgs::Spread {
                m_cubes: problem_count.x,
                n_cubes: problem_count.y,
                batch_cubes: problem_count.z,
            },
        }
    }
}

/// Heuristic algorithm to factor the total number of cubes into (x, y, z) dimensions
/// such that no dimension surpasses its maximum.
fn spread_cube_count_plan(
    problem_count: Count3d,
    max_x: u32,
    max_y: u32,
    max_z: u32,
) -> CubeCountPlanKind {
    let mut best = None;

    let mut z = max_z;
    while z >= 1 {
        let xy_cubes = problem_count.total().div_ceil(z as u64);

        let mut y = max_y;
        while y >= 1 {
            let x64 = xy_cubes.div_ceil(y as u64);
            if x64 <= max_x as u64 {
                let x = x64 as u32;
                let volume = x as u64 * y as u64 * z as u64;
                let score = (volume, std::cmp::Reverse(z), std::cmp::Reverse(y));

                if best.is_none_or(|(_, _, _, _, best_score)| score < best_score) {
                    best = Some((x, y, z, volume, score));
                }
            }

            if y == 1 {
                break;
            }
            y /= 2;
        }

        if z == 1 {
            break;
        }
        z /= 2;
    }

    if let Some((x, y, z, _, _)) = best {
        CubeCountPlanKind::Spread {
            problem_count,
            spread_count: Count3d { x, y, z },
        }
    } else {
        panic!("No valid cube spread plan")
    }
}
