//! Inferred-blueprint smoke tests for plane-accelerated routines.
//!
//! One test per (routine, backend) variant exercises the selector's heuristic
//! against a representative shape; that is enough to catch selector regressions
//! without blowing up compile time.

use cubecl::prelude::*;
use cubek_matmul::{
    definition::MatmulElems,
    multi_level::{Strategy as MultiLevel, definition::adjust_dtypes},
};
use cubek_test_utils::{ScopeUnderTest, TestOutcome, ValidationResult};

use crate::matmul::{ByStrategy, Declines, F16, F32, Product};

#[test]
fn simple_cyclic_cmma() {
    let routine = ByStrategy::new(
        MultiLevel::SimpleCyclicCmma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

/// A small f32 problem runs rather than declining: a routine that runs at 37 rows also runs at 32,
/// since autotune buckets the two together. Asserted as executed on a device with cmma, where the
/// default test policy would accept the decline; reported as skipped on one without. A device
/// can offer cmma for narrower types only: Vulkan on NVIDIA has f16 and bf16 but no f32 or tf32.
#[test]
fn simple_cyclic_cmma_runs_a_small_f32_problem() {
    let client = cubecl::test_device().client();
    let mut dtypes = MatmulElems::from_single_dtype(f32::elem_type_native());
    adjust_dtypes(&client, &mut dtypes, true);
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == dtypes.lhs_register
            && cfg.b_type == dtypes.rhs_register
            && cfg.cd_type == dtypes.acc_register
    });
    if !offered {
        TestOutcome::Validated(ValidationResult::Skipped(
            "the device offers no cmma instruction for an f32 matmul".to_string(),
        ))
        .enforce();
        return;
    }
    let cmma = ByStrategy::new(
        MultiLevel::SimpleCyclicCmma(Default::default()),
        Declines::Never,
    );
    cmma.check([Product::of(F32).m(32).n(32).k(64).in_eighths()]);
}

#[test]
fn simple_cyclic_mma() {
    let routine = ByStrategy::new(
        MultiLevel::SimpleCyclicMma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn double_cyclic_cmma() {
    let routine = ByStrategy::new(
        MultiLevel::DoubleCyclicCmma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn double_cyclic_mma() {
    let routine = ByStrategy::new(
        MultiLevel::DoubleCyclicMma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn specialized_cyclic_cmma() {
    let routine = ByStrategy::new(
        MultiLevel::SpecializedCyclicCmma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn specialized_cyclic_mma() {
    let routine = ByStrategy::new(
        MultiLevel::SpecializedCyclicMma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

/// The specialized routine loads on its load-only planes whatever load flows a forced blueprint
/// carries. Regression for a blueprint left at the default flows: the cube was sized for
/// load-only planes while the stages and readers expected the compute planes to load, so part of
/// every stage was never loaded. Two compute planes per stage leave a stage the load-only plane
/// covers only if the flows agree.
#[test]
fn specialized_cyclic_cmma_runs_a_forced_blueprint_with_default_load_flows() {
    use cubek_matmul::{
        multi_level::{
            PartitionSize, StageSize,
            components::{stage::PartitionBuffering, tile::TileMatmulKind},
            definition::{BatchMatmulBlueprint, TilingScheme},
        },
        routine::BlueprintStrategy,
    };

    let client = cubecl::test_device().client();
    let product = Product::of(F16).m(256).n(256).k(256);
    let tiling_scheme = TilingScheme::builder()
        .with_tile_size((16, 16, 16).into())
        .with_partition_size(PartitionSize { m: 1, n: 1, k: 1 })
        .with_stage_size(StageSize { m: 2, n: 1, k: 1 })
        .build()
        .unwrap();
    let blueprint = BatchMatmulBlueprint::builder(
        TileMatmulKind::Cmma,
        tiling_scheme,
        client.properties().hardware.plane_size_max,
        &product.problem(),
    )
    .partition_buffering(PartitionBuffering::Single)
    .build();
    let forced = ByStrategy::new(
        MultiLevel::SpecializedCyclicCmma(BlueprintStrategy::Forced(blueprint)),
        Declines::AsItStates,
    );
    forced.check([product]);
}

#[test]
fn ordered_double_cmma() {
    let routine = ByStrategy::new(
        MultiLevel::OrderedDoubleCmma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn ordered_double_mma() {
    let routine = ByStrategy::new(
        MultiLevel::OrderedDoubleMma(Default::default()),
        Declines::AsItStates,
    );
    routine.check([Product::of(F16).m(256).n(256).k(256)]);
}

#[test]
fn simple_cyclic_cmma_small_k_and_partial_line_width_rows_stay_correct() {
    let cmma = ByStrategy::new(
        MultiLevel::SimpleCyclicCmma(Default::default()),
        Declines::AsItStates,
    );
    let products = [
        Product::of(F32).m(382).n(10).k(1).in_eighths(),
        Product::of(F32).m(382).n(10).k(3).in_eighths(),
        Product::of(F32).m(4096).n(14).k(5).in_eighths(),
    ];
    for _ in 0..20 {
        cmma.check(products.clone());
    }
}
