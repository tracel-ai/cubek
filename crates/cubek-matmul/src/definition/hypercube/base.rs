use crate::definition::hypercube::{
    CubeCountPlanBlueprint, GlobalOrder, GlobalOrderBlueprint,
    cube_count_plan::{CubeCountPlan, CubeCountPlanConfig},
};
use crate::definition::{MatmulProblem, MatmulSetupError, TilingScheme};
use cubecl::CubeCount;

#[derive(Debug, Clone)]
/// Determines how to launch the hypercube, i.e. anything
/// relevant to CubeCount and where a Cube at a cube position should work
pub struct HypercubeBlueprint {
    pub cube_span: CubeSpan,
    pub global_order: GlobalOrder,
    pub cube_count_plan_selection: CubeCountPlanBlueprint,
}

/// Builder for creating a [HypercubeBlueprint]
pub struct HypercubeBlueprintBuilder<'a> {
    tiling_scheme: &'a TilingScheme,
    global_order: GlobalOrderBlueprint,
    cube_count_plan_config: Option<CubeCountPlanBlueprint>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Determines how to launch the hypercube, i.e. anything
/// relevant to CubeCount and where a Cube at a cube position should work
/// Similar to HypercubeBlueprint but injected in kernel as comptime struct
pub struct HypercubeConfig {
    pub cube_span: CubeSpan,
    pub global_order: GlobalOrder,
    pub cube_count_plan_blueprint: CubeCountPlanConfig,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// Number of elements each cube covers in the tensors
pub struct CubeSpan {
    pub m: u32,
    pub n: u32,
    pub batch: u32,
}

impl HypercubeBlueprint {
    /// Create a builder for HypercubeBlueprint
    pub fn builder<'a>(tiling_scheme: &'a TilingScheme) -> HypercubeBlueprintBuilder<'a> {
        HypercubeBlueprintBuilder::new(tiling_scheme)
    }

    pub(crate) fn to_hypercube_config(
        &self,
        problem: &MatmulProblem,
        max_cube_count: CubeCount,
    ) -> HypercubeConfig {
        let cube_count_plan = CubeCountPlan::from_selection(self, problem, max_cube_count);
        let cube_count_plan_config = CubeCountPlanConfig::from_cube_count_plan(cube_count_plan);

        HypercubeConfig {
            cube_span: self.cube_span,
            global_order: self.global_order,
            cube_count_plan_blueprint: cube_count_plan_config,
        }
    }
}

impl HypercubeConfig {
    /// Returns an error if:
    /// - The global order is swizzle but its assumptions are not met
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

impl<'a> HypercubeBlueprintBuilder<'a> {
    fn new(tiling_scheme: &'a TilingScheme) -> Self {
        Self {
            tiling_scheme,
            global_order: GlobalOrderBlueprint::default(),
            cube_count_plan_config: None,
        }
    }

    /// Set the [GlobalOrderBlueprint]
    pub fn global_order(mut self, global_order: GlobalOrderBlueprint) -> Self {
        self.global_order = global_order;
        self
    }

    /// Set the [CubeCountPlanBlueprint]
    pub fn cube_count_plan(mut self, cube_count_plan_config: CubeCountPlanBlueprint) -> Self {
        self.cube_count_plan_config = Some(cube_count_plan_config);
        self
    }

    /// Build the HypercubeBlueprint
    pub fn build(self) -> HypercubeBlueprint {
        let cube_span = CubeSpan {
            m: self.tiling_scheme.elements_per_global_partition_along_m(),
            n: self.tiling_scheme.elements_per_global_partition_along_n(),
            batch: self.tiling_scheme.global_partition_size.batches,
        };

        let global_order = self.global_order.into_order(&cube_span);
        let cube_pos_strategy = self.cube_count_plan_config.unwrap_or_default();

        HypercubeBlueprint {
            cube_span,
            global_order,
            cube_count_plan_selection: cube_pos_strategy,
        }
    }
}

impl HypercubeConfig {
    /// Make a CubeCountPlan from the problem, constrained to not exceed the maximal cube count
    pub fn cube_count_plan(
        &self,
        problem: &MatmulProblem,
        max_cube_count: &CubeCount,
    ) -> CubeCountPlan {
        CubeCountPlan::from_blueprint(self, problem, max_cube_count)
    }
}
