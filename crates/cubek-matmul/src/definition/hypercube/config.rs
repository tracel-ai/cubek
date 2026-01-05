use crate::definition::MatmulProblem;
use crate::definition::hypercube::GlobalOrder;
use crate::definition::hypercube::base::CubeSpan;
use cubecl::CubeCount;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Determines how to launch the hypercube, i.e. anything
/// relevant to CubeCount and where a Cube at a cube position should work
/// Similar to HypercubeBlueprint but injected in kernel as comptime struct
pub struct HypercubeConfig {
    pub cube_span: CubeSpan,
    pub global_order: GlobalOrder,
    pub cube_count_plan_config: CubeCountPlanConfig,
}

impl HypercubeConfig {
    /// Make a CubeCountPlan from the problem, constrained to not exceed the maximal cube count
    pub fn cube_count_plan(
        &self,
        problem: &MatmulProblem,
        max_cube_count: &CubeCount,
    ) -> CubeCountPlan {
        CubeCountPlan::from_config(self, problem, max_cube_count)
    }
}
