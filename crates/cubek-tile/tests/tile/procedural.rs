//! Procedural tiles evaluate logical coordinates without a tensor-backed source.

use cubecl::{Runtime, TestRuntime, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const REDUCE: Axis = Axis(2);

#[cube(launch)]
fn procedural_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] recipe: ProceduralRecipe,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural(comptime!(space.clone()), recipe);
    // Select the second 2x3 tile, then evaluate its first logical coordinate. This exercises
    // `Tile::at`'s origin rebasing rather than merely the top-level recipe evaluation.
    let region = Region::trailing(comptime!(space.clone()), 1usize, 1usize);
    let source = source.at(&region);
    let mut pos = Coords::<u32>::new();
    pos.push(0u32.runtime());
    pos.push(0u32.runtime());

    let mut output = output.tile(space);
    output.init(source.procedural_value(pos));
}

/// Exercise a staged schedule whose procedural operand remains coordinate-backed in place.
#[cube(launch)]
fn procedural_stage_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] recipe: ProceduralRecipe,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural(comptime!(space.clone()), recipe);
    let output = output.tile(comptime!(space.clone()));
    let mut ring = Ring::unary(
        &source,
        comptime!(space.clone()),
        comptime!(space.clone()),
        1usize,
    );
    let walk = Walk::over(source.runtime_space());
    for region in walk {
        let staging = ring.slot_mut(0usize);
        staging.fill_streamed(&source, &region);
        staging.consume_final(|staged| {
            output.at(&region).copy_from(staged);
        });
    }
}

/// A staged contraction with an in-place procedural lhs and a materialized tensor rhs.
#[cube(launch)]
fn procedural_mma_kernel<E: Float>(
    rhs: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let lhs = Tile::<E>::procedural(
        comptime!(space.project(&[ROW, REDUCE])),
        comptime!(ProceduralRecipe::axis_index(REDUCE)),
    );
    let rhs = rhs.tile(comptime!(space.clone()));
    let mut output = output.tile(space);
    output.zero();
    output.mma(&lhs, &rhs);
}

fn run(recipe: ProceduralRecipe) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = Tiling::new()
        .extents(&[(ROW, 4), (COL, 6)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(3))
        })
        .build();
    let output = TestInput::builder(client.clone(), shape![4, 6])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    procedural_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        recipe,
        dtype,
    );

    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

fn run_copy(recipe: ProceduralRecipe) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = Tiling::new()
        .extents(&[(ROW, 4), (COL, 6)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(3))
        })
        .build();
    let output = TestInput::builder(client.clone(), shape![4, 6])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    procedural_stage_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        recipe,
        dtype,
    );

    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

fn run_mma() -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = Tiling::new()
        .extents(&[(ROW, 4), (COL, 4), (REDUCE, 4)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(2))
                .axis(REDUCE, Cut::sequential(2))
        })
        .build();
    let rhs = TestInput::builder(client.clone(), shape![4, 4])
        .dtype(dtype)
        .arange()
        .generate_without_host_data();
    let output = TestInput::builder(client.clone(), shape![4, 4])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    procedural_mma_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        TileArgLaunch::new(
            rhs.binding().into_tensor_arg(),
            // Only the tensor operand takes a stage; the procedural lhs stays coordinate-backed
            // at its default all-in-place residence, which is the point of the test.
            TileSpec::direct(&[REDUCE, COL]).residence(&[Residence::Auto]),
        ),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        dtype,
    );

    HostData::from_tensor_handle(&client, output, HostDataType::F32)
}

#[test]
fn evaluates_separable_axis_products_after_region_rebase() {
    // The selected region begins at (2, 3), so AxisIndex(ROW) * AxisIndex(COL) is 6.
    let got = run(ProceduralRecipe::axis_product(vec![
        ProceduralRecipe::axis_index(ROW),
        ProceduralRecipe::axis_index(COL),
    ]));
    for row in 0..4 {
        for col in 0..6 {
            assert_eq!(got.get_f32(&[row, col]), 6.0);
        }
    }
}

#[test]
fn staged_buffering_keeps_coordinate_varying_values_in_place() {
    let got = run_copy(ProceduralRecipe::axis_product(vec![
        ProceduralRecipe::axis_index(ROW),
        ProceduralRecipe::axis_index(COL),
    ]));
    for row in 0..4 {
        for col in 0..6 {
            assert_eq!(got.get_f32(&[row, col]), (row * col) as f32);
        }
    }
}

#[test]
fn staged_mma_materializes_only_the_tensor_operand() {
    let got = run_mma();
    for row in 0..4 {
        for col in 0..4 {
            let expected = (0..4).map(|k| (k * (k * 4 + col)) as f32).sum::<f32>();
            assert_eq!(got.get_f32(&[row, col]), expected);
        }
    }
}

#[test]
fn evaluates_zero_one_and_uniform_recipes() {
    for (recipe, expected) in [
        (ProceduralRecipe::zero(), 0.0),
        (ProceduralRecipe::one(), 1.0),
        (ProceduralRecipe::uniform(4), 0.25),
    ] {
        let got = run(recipe);
        assert_eq!(got.get_f32(&[0, 0]), expected);
    }
}
