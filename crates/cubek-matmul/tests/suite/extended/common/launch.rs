use super::*;
use crate::suite::launcher::test_matmul_algorithm;
use cubecl::Runtime;
use cubecl::TestRuntime;
use cubek_matmul::definition::{MatmulElems, TilingBlueprint, TilingScheme};
use cubek_matmul::routines::BlueprintStrategy;

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

    test_matmul_algorithm::<Algorithm>(
        client,
        problem,
        BlueprintStrategy::Forced(blueprint),
        input_representation(),
    );
}
