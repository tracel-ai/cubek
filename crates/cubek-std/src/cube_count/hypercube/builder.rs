use crate::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};

/// Builder for creating a [HypercubeBlueprint]
pub struct HypercubeBlueprintBuilder {
    global_order: Option<GlobalOrder>,
    cube_count_strategy: Option<CubeCountStrategy>,
}

impl HypercubeBlueprintBuilder {
    pub(crate) fn new() -> Self {
        Self {
            global_order: None,
            cube_count_strategy: None,
        }
    }

    /// Set the [GlobalOrder]
    pub fn global_order(mut self, global_order: GlobalOrder) -> Self {
        self.global_order = Some(global_order);
        self
    }

    /// Set the [CubeCountStrategy]
    pub fn cube_count_strategy(mut self, cube_count_strategy: CubeCountStrategy) -> Self {
        self.cube_count_strategy = Some(cube_count_strategy);
        self
    }

    /// Build the HypercubeBlueprint
    pub fn build(self) -> HypercubeBlueprint {
        HypercubeBlueprint {
            global_order: self.global_order.unwrap_or_default(),
            cube_count_strategy: self.cube_count_strategy.unwrap_or_default(),
        }
    }
}
