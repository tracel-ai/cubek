use cubecl::{CubeCount, Runtime, prelude::ScalarArg};

use crate::definition::{
    GlobalOrder, MatmulProblem, SmAllocation, TilingScheme,
    hypercube::{
        blueprint::HypercubeBlueprint,
        cube_count::{
            CubeCountStrategy,
            mapping::{CubeMappingLaunch, CubeMappingStrategyArgs},
        },
    },
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CubeCountPlan {
    pub global_order: GlobalOrder,
    pub kind: CubeCountPlanKind,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CubeCountPlanKind {
    FromProblem {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Sm {
        cubes_first: bool,
        num_sms_used: u32,
        cubes_per_sm: u32,
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
        num_sms: u32,
        sm_usage: SmAllocation,
    },
    Flattened {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
    },
    Spread {
        m_cubes: u32,
        n_cubes: u32,
        batch_cubes: u32,
        x: u32,
        y: u32,
        z: u32,
    },
}

impl CubeCountPlan {
    // Will check if the wanted cube count plan is possible, otherwise will fallback to spread
    pub fn from_blueprint(
        blueprint: &HypercubeBlueprint,
        tiling_scheme: &TilingScheme,
        problem: &MatmulProblem,
        max_cube_count: &(u32, u32, u32),
    ) -> CubeCountPlan {
        let (max_x, max_y, max_z) = *max_cube_count;

        let m_cubes =
            (problem.m as u32).div_ceil(tiling_scheme.elements_per_global_partition_along_m());
        let n_cubes =
            (problem.n as u32).div_ceil(tiling_scheme.elements_per_global_partition_along_n());
        let batch_cubes =
            (problem.num_batches() as u32).div_ceil(tiling_scheme.global_partition_size.batches);

        let plan_kind = match blueprint.cube_count_strategy {
            CubeCountStrategy::FromProblem => {
                if m_cubes > max_x || n_cubes > max_y || batch_cubes > max_z {
                    None
                } else {
                    Some(CubeCountPlanKind::FromProblem {
                        m_cubes,
                        n_cubes,
                        batch_cubes,
                    })
                }
            }
            CubeCountStrategy::Sm {
                cubes_first,
                num_sms,
                sm_usage,
            } => {
                let (num_sms_used, cubes_per_sm) =
                    sm_usage.allocate(num_sms, m_cubes * n_cubes * batch_cubes);

                if (cubes_per_sm >= if cubes_first { max_x } else { max_y })
                    || (num_sms_used >= if cubes_first { max_y } else { max_x })
                {
                    None
                } else {
                    Some(CubeCountPlanKind::Sm {
                        cubes_first,
                        num_sms_used,
                        cubes_per_sm,
                        m_cubes,
                        n_cubes,
                        batch_cubes,
                        num_sms,
                        sm_usage,
                    })
                }
            }
            CubeCountStrategy::Flattened => {
                if m_cubes * n_cubes * batch_cubes >= max_x {
                    None
                } else {
                    Some(CubeCountPlanKind::Flattened {
                        m_cubes,
                        n_cubes,
                        batch_cubes,
                    })
                }
            }
            CubeCountStrategy::Spread => None,
        };

        CubeCountPlan {
            global_order: blueprint.global_order,
            kind: plan_kind.unwrap_or_else(|| {
                spread_cube_count_plan(m_cubes, n_cubes, batch_cubes, max_x, max_y, max_z)
            }),
        }
    }

    pub fn new_from_problem(m_cubes: u32, n_cubes: u32, batch_cubes: u32) -> Self {
        Self {
            global_order: Default::default(),
            kind: CubeCountPlanKind::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            },
        }
    }

    pub fn can_yield_extra_cubes(&self) -> bool {
        self.kind.can_yield_extra_cubes()
    }

    pub fn resolve(&self) -> CubeCount {
        self.kind.resolve()
    }

    pub fn as_args<'a, R: Runtime>(&self) -> CubeMappingLaunch<'a, R> {
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
                m_cubes,
                n_cubes,
                batch_cubes,
                ..
            } => num_sms_used * cubes_per_sm != m_cubes * n_cubes * batch_cubes,

            CubeCountPlanKind::Spread {
                m_cubes,
                n_cubes,
                batch_cubes,
                x,
                y,
                z,
            } => m_cubes * n_cubes * batch_cubes != x * y * z,
        }
    }

    fn resolve(&self) -> CubeCount {
        match self {
            CubeCountPlanKind::FromProblem {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(*m_cubes, *n_cubes, *batch_cubes),

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

            CubeCountPlanKind::Flattened {
                m_cubes,
                n_cubes,
                batch_cubes,
            } => CubeCount::Static(m_cubes * n_cubes * batch_cubes, 1, 1),

            CubeCountPlanKind::Spread { x, y, z, .. } => CubeCount::Static(*x, *y, *z),
        }
    }

    fn mapping_strategy<'a, R: Runtime>(&self) -> CubeMappingStrategyArgs<'a, R> {
        match self {
            CubeCountPlanKind::FromProblem { .. } => CubeMappingStrategyArgs::FromProblem,

            CubeCountPlanKind::Sm {
                cubes_first,
                m_cubes,
                n_cubes,
                batch_cubes,
                ..
            } => {
                if *cubes_first {
                    CubeMappingStrategyArgs::CubeFirst {
                        m_cubes: ScalarArg::new(*m_cubes),
                        n_cubes: ScalarArg::new(*n_cubes),
                        batch_cubes: ScalarArg::new(*batch_cubes),
                    }
                } else {
                    CubeMappingStrategyArgs::SmFirst {
                        m_cubes: ScalarArg::new(*m_cubes),
                        n_cubes: ScalarArg::new(*n_cubes),
                        batch_cubes: ScalarArg::new(*batch_cubes),
                    }
                }
            }

            CubeCountPlanKind::Flattened {
                m_cubes, n_cubes, ..
            } => CubeMappingStrategyArgs::Flattened {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
            },

            CubeCountPlanKind::Spread {
                m_cubes,
                n_cubes,
                batch_cubes,
                ..
            } => CubeMappingStrategyArgs::Spread {
                m_cubes: ScalarArg::new(*m_cubes),
                n_cubes: ScalarArg::new(*n_cubes),
                batch_cubes: ScalarArg::new(*batch_cubes),
            },
        }
    }
}

/// Heuristic algorithm to factor the total number of cubes into (x, y, z) dimensions
/// such that no dimension surpasses its maximum.
fn spread_cube_count_plan(
    m_cubes: u32,
    n_cubes: u32,
    batch_cubes: u32,
    max_x: u32,
    max_y: u32,
    max_z: u32,
) -> CubeCountPlanKind {
    let total_cubes = m_cubes * n_cubes * batch_cubes;

    let mut best = None;

    let mut z = max_z;
    while z >= 1 {
        let xy_cubes = total_cubes.div_ceil(z);

        let mut y = max_y;
        while y >= 1 {
            let x = xy_cubes.div_ceil(y);
            if x <= max_x {
                let volume = x * y * z;
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
            m_cubes,
            n_cubes,
            batch_cubes,
            x,
            y,
            z,
        }
    } else {
        panic!("No valid cube spread plan")
    }
}
