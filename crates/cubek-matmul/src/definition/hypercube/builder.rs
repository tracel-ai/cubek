use crate::definition::{
    GlobalOrderStrategy, TilingScheme,
    hypercube::{blueprint::HypercubeBlueprint, cube_count::CubeCountStrategy},
};

/// Builder for creating a [HypercubeBlueprint]
pub struct HypercubeBlueprintBuilder<'a> {
    tiling_scheme: &'a TilingScheme,
    global_order_strategy: GlobalOrderStrategy,
    cube_count_strategy: Option<CubeCountStrategy>,
}

impl<'a> HypercubeBlueprintBuilder<'a> {
    pub(crate) fn new(tiling_scheme: &'a TilingScheme) -> Self {
        Self {
            tiling_scheme,
            global_order_strategy: GlobalOrderStrategy::default(),
            cube_count_strategy: None,
        }
    }

    /// Set the [GlobalOrderStrategy]
    pub fn global_order_strategy(mut self, global_order_strategy: GlobalOrderStrategy) -> Self {
        self.global_order_strategy = global_order_strategy;
        self
    }

    /// Set the [CubeCountStrategy]
    pub fn cube_count_strategy(mut self, cube_count_plan_blueprint: CubeCountStrategy) -> Self {
        self.cube_count_strategy = Some(cube_count_plan_blueprint);
        self
    }

    /// Build the HypercubeBlueprint
    pub fn build(self) -> HypercubeBlueprint {
        let global_order = self.global_order_strategy.into_order(self.tiling_scheme);
        let cube_count_strategy = self.cube_count_strategy.unwrap_or_default();

        HypercubeBlueprint {
            global_order,
            cube_count_strategy,
        }
    }
}
