use cubecl::CubeCount;

use crate::definition::{
    GlobalOrder, MatmulProblem, MatmulSetupError, TilingScheme,
    hypercube::{base::CubeSpan, builder::HypercubeBlueprintBuilder},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Determines how to launch the hypercube, i.e. anything
/// relevant to CubeCount and where a Cube at a cube position should work
pub struct HypercubeBlueprint {
    pub cube_span: CubeSpan,
    pub global_order: GlobalOrder,
    pub cube_count_plan_blueprint: CubeCountPlanBlueprint,
}

impl HypercubeBlueprint {
    /// Create a builder for HypercubeBlueprint
    pub fn builder<'a>(tiling_scheme: &'a TilingScheme) -> HypercubeBlueprintBuilder<'a> {
        HypercubeBlueprintBuilder::new(tiling_scheme)
    }

    pub(crate) fn to_hypercube_config(
        &self,
        // this is bad. we don't know m,n,total_batches when turning blueprint into config (comptime inside kernel)
        m: u32,
        n: u32,
        total_batches: u32,
        max_cube_count: CubeCount,
    ) -> HypercubeConfig {
        let cube_count_plan =
            CubeCountPlan::from_blueprint(self, m, n, total_batches, &max_cube_count);
        let cube_count_plan_config = CubeCountPlanConfig::from_cube_count_plan(cube_count_plan);

        HypercubeConfig {
            cube_span: self.cube_span,
            global_order: self.global_order,
            cube_count_plan_config,
        }
    }

    /// Returns an error if:
    /// - The global order is swizzle but its assumptions are not met
    // TODO don't forget to call (in validate_blueprint likely)
    pub fn validate(&self, problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        let m_cubes = (problem.m as u32).div_ceil(self.cube_span.m);
        let n_cubes = (problem.n as u32).div_ceil(self.cube_span.n);

        use GlobalOrder::*;

        match self.global_order {
            RowMajor | ColMajor => Ok(()),

            SwizzleRowMajor(w) if !m_cubes.is_multiple_of(w) => {
                Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "In swizzle row major, number of cubes in m {m_cubes:?} must be divisible by swizzle step length {w:?}."
                ))))
            }

            SwizzleColMajor(w) if !n_cubes.is_multiple_of(w) => {
                Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "In swizzle col major, number of cubes in n {n_cubes:?} must be divisible by swizzle step length {w:?}."
                ))))
            }

            _ => Ok(()),
        }
    }
}
