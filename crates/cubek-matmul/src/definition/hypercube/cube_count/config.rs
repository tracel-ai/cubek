// use crate::definition::{SmAllocation, hypercube::cube_count::CubeCountPlan};

// #[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// /// Config derived from CubeCountPlan to be used comptime in kernels
// pub enum CubeCountConfig {
//     FromProblem,

//     Sm {
//         cubes_first: bool,
//         num_sms: u32,
//         sm_usage: SmAllocation,
//         can_yield_extra_cubes: bool,
//     },

//     Flattened,

//     Spread {
//         can_yield_extra_cubes: bool,
//     },
// }

// impl CubeCountConfig {
//     /// Whether the CubeCount will have more cubes than strictly necessary.
//     pub fn can_yield_extra_cubes(&self) -> bool {
//         match self {
//             CubeCountConfig::FromProblem | CubeCountConfig::Flattened => false,
//             CubeCountConfig::Sm {
//                 can_yield_extra_cubes,
//                 ..
//             } => *can_yield_extra_cubes,
//             CubeCountConfig::Spread {
//                 can_yield_extra_cubes,
//             } => *can_yield_extra_cubes,
//         }
//     }

//     pub(crate) fn from_cube_count_plan(cube_count_plan: CubeCountPlan) -> CubeCountConfig {
//         match cube_count_plan {
//             CubeCountPlan::FromProblem { .. } => CubeCountConfig::FromProblem,
//             CubeCountPlan::Sm {
//                 cubes_first,
//                 num_sms,
//                 sm_usage,
//                 ..
//             } => CubeCountConfig::Sm {
//                 cubes_first,
//                 num_sms,
//                 sm_usage,
//                 can_yield_extra_cubes: cube_count_plan.can_yield_extra_cubes(),
//             },
//             CubeCountPlan::Flattened { .. } => CubeCountConfig::Flattened,
//             CubeCountPlan::Spread { .. } => CubeCountConfig::Spread {
//                 can_yield_extra_cubes: cube_count_plan.can_yield_extra_cubes(),
//             },
//         }
//     }
// }
