use super::*;
use cubecl::{Runtime, TestRuntime};
use cubek_matmul::definition::{TilingBlueprint, TilingScheme};

#[test]
pub fn test() {
    let client = TestRuntime::client(&Default::default());

    let tiling_scheme = stage(partition(tile_size(TilingScheme::builder())))
        .build()
        .unwrap();
    let plane_dim = client.properties().hardware.plane_size_max;
    let problem = problem();
    let blueprint_builder = TilingBlueprint::builder(tiling_scheme, plane_dim, &problem);
    let blueprint = blueprint_builder
        .shared_swizzle(swizzle())
        .hypercube_blueprint(hypercube_blueprint(&tiling_scheme, &problem))
        .partition_buffering(partition_buffering())
        .load_specialization_config(specialization())
        .build();

    launch(client, problem, blueprint);
}
