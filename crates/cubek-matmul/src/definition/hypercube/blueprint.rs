use crate::definition::{
    GlobalOrder, TilingScheme,
    hypercube::{builder::HypercubeBlueprintBuilder, cube_count::CubeCountStrategy},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Determines how to launch the hypercube, i.e. anything
/// relevant to CubeCount and where a Cube at a cube position should work
pub struct HypercubeBlueprint {
    pub global_order: GlobalOrder,
    pub cube_count_strategy: CubeCountStrategy,
}

impl HypercubeBlueprint {
    /// Create a builder for HypercubeBlueprint
    pub fn builder<'a>(tiling_scheme: &'a TilingScheme) -> HypercubeBlueprintBuilder<'a> {
        HypercubeBlueprintBuilder::new(tiling_scheme)
    }
}
