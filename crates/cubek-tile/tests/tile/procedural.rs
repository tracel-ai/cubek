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

/// Exercise a staged schedule whose procedural operand remains coordinate-backed in place. Such an
/// operand is never filled, so the slot hands out the recipe whole and this read selects the
/// region out of it, which is what a lowered walk's `read_operand` does for the same payload.
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
        staging.publish();
        staging.consume(|staged| {
            output.at(&region).copy_from(&staged.at(&region));
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

fn run_mma(buffering: Buffering) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = Tiling::new()
        .extents(&[(ROW, 4), (COL, 4), (REDUCE, 4)])
        .level(WalkOrder::RowMajor, buffering, |level| {
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
            TileSpec::direct(&[REDUCE, COL]).residence(&[Residence::Smem]),
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

/// Exercise a staged schedule whose procedural operand is cooperatively staged into shared memory.
#[cube(launch)]
fn procedural_smem_stage_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] recipe: ProceduralRecipe,
    #[define(E)] _dtype: ElemType,
) {
    let stage = comptime!(StagePlan::new(&[Residence::Smem], StageStorage::Strided, 0));
    let source = Tile::<E>::procedural_resident(comptime!(space.clone()), recipe, stage);
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
        staging.publish();
        staging.consume(|staged| {
            output.at(&region).copy_from(staged);
        });
    }
}

/// Exercise direct Tile::copy_from from a procedural tile to a global memory window.
#[cube(launch)]
fn procedural_direct_copy_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] recipe: ProceduralRecipe,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural(comptime!(space.clone()), recipe);
    let mut output = output.tile(space);
    output.copy_from(&source);
}

/// Descend once before reading: `Space::divide` makes child extents static, while a procedural
/// bound must retain the parent axis positions that were dynamic at construction.
#[cube(launch)]
fn procedural_divided_copy_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] recipe: ProceduralRecipe,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural(comptime!(space.clone()), recipe);
    let region = Region::trailing(comptime!(space.clone()), 0usize, 0usize);
    let source = source.at(&region);
    let output = output.tile(comptime!(space.clone()));
    let mut output = output.at(&region);
    output.copy_from(&source);
}

fn run_smem_stage(recipe: ProceduralRecipe) -> HostData {
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

    procedural_smem_stage_kernel::launch::<TestRuntime>(
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

fn run_direct_copy(recipe: ProceduralRecipe) -> HostData {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = Space::new(&[(ROW, 4), (COL, 6)]);
    let output = TestInput::builder(client.clone(), shape![4, 6])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    procedural_direct_copy_kernel::launch::<TestRuntime>(
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
fn stages_coordinate_varying_values_through_shared_memory() {
    let got = run_smem_stage(ProceduralRecipe::axis_product(vec![
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
fn copies_procedural_directly_to_global_memory() {
    let got = run_direct_copy(ProceduralRecipe::axis_product(vec![
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
fn dynamic_procedural_axis_does_not_require_a_source_bound() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let concrete = Space::new(&[(ROW, 4), (COL, 6)]);
    let space = concrete.clone().with_dynamic(&[ROW]);
    let output = TestInput::builder(client.clone(), shape![4, 6])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // `Tile::procedural` receives the dynamic kernel space directly. Unlike the output tile,
    // it cannot witness ROW's runtime extent, so construction must leave that axis unmasked.
    procedural_direct_copy_kernel::launch::<TestRuntime>(
        &client,
        concrete.cube_count(),
        concrete.cube_dim(&client),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        ProceduralRecipe::one(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..4 {
        for col in 0..6 {
            assert_eq!(got.get_f32(&[row, col]), 1.0);
        }
    }
}

#[test]
fn dynamic_axis_keeps_static_procedural_bound_aligned_after_divide() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let concrete = Tiling::new()
        .extents(&[(ROW, 4), (COL, 6)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(4))
        })
        .build();
    let space = concrete.clone().with_dynamic(&[ROW]);
    let output = TestInput::builder(client.clone(), shape![4, 6])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    procedural_divided_copy_kernel::launch::<TestRuntime>(
        &client,
        concrete.cube_count(),
        concrete.cube_dim(&client),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        ),
        space,
        ProceduralRecipe::one(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..4 {
        for col in 0..6 {
            let expected = if row < 2 && col < 4 { 1.0 } else { 0.0 };
            assert_eq!(got.get_f32(&[row, col]), expected);
        }
    }
}

#[test]
fn staged_mma_materializes_only_the_tensor_operand() {
    let got = run_mma(Buffering::SINGLE);
    for row in 0..4 {
        for col in 0..4 {
            let expected = (0..4).map(|k| (k * (k * 4 + col)) as f32).sum::<f32>();
            assert_eq!(got.get_f32(&[row, col]), expected);
        }
    }
}

#[test]
fn double_buffered_mma_with_in_place_procedural_operand() {
    let got = run_mma(Buffering::DOUBLE);
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
