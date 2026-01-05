use crate::definition::{
    GlobalOrderDefinition, TilingScheme,
    hypercube::{base::CubeSpan, blueprint::HypercubeBlueprint},
};

/// Builder for creating a [HypercubeBlueprint]
pub struct HypercubeBlueprintBuilder<'a> {
    tiling_scheme: &'a TilingScheme,
    global_order_definition: GlobalOrderDefinition,
    cube_count_plan_blueprint: Option<CubeCountPlanBlueprint>,
}

impl<'a> HypercubeBlueprintBuilder<'a> {
    pub(crate) fn new(tiling_scheme: &'a TilingScheme) -> Self {
        Self {
            tiling_scheme,
            global_order_definition: GlobalOrderDefinition::default(),
            cube_count_plan_blueprint: None,
        }
    }

    /// Set the [GlobalOrderBlueprint]
    pub fn global_order(mut self, global_order_definition: GlobalOrderDefinition) -> Self {
        self.global_order_definition = global_order_definition;
        self
    }

    /// Set the [CubeCountPlanBlueprint]
    pub fn cube_count_plan_blueprint(
        mut self,
        cube_count_plan_blueprint: CubeCountPlanBlueprint,
    ) -> Self {
        self.cube_count_plan_blueprint = Some(cube_count_plan_blueprint);
        self
    }

    /// Build the HypercubeBlueprint
    pub fn build(self) -> HypercubeBlueprint {
        let cube_span = CubeSpan {
            m: self.tiling_scheme.elements_per_global_partition_along_m(),
            n: self.tiling_scheme.elements_per_global_partition_along_n(),
            batch: self.tiling_scheme.global_partition_size.batches,
        };

        let global_order = self.global_order_definition.into_order(&cube_span);
        let cube_pos_strategy = self.cube_count_plan_blueprint.unwrap_or_default();

        HypercubeBlueprint {
            cube_span,
            global_order,
            cube_count_plan_blueprint: cube_pos_strategy,
        }
    }
}
