//! Procedural tiles accept user-defined recipes.

use core::f32::consts::PI;

use cubecl::{Runtime, TestRuntime, prelude::*, std::tensor::TensorHandle, zspace::shape};
use cubecl_common::{ComptimeFloat, Ratio};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::*;

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 4;
const COLS: usize = 6;

#[derive(CubeType, Clone)]
struct ProductAxes {
    #[cube(comptime)]
    row: Axis,
    #[cube(comptime)]
    col: Axis,
}

impl RecipeMeta for ProductAxes {}

#[cube]
impl<T: Float> Recipe<T> for ProductAxes {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        let row = coordinates.along(self.row);
        let col = coordinates.along(self.col);
        T::cast_from(row) * T::cast_from(col)
    }
}

#[derive(CubeType, Clone)]
struct AxisValue {
    #[cube(comptime)]
    axis: Axis,
    scale: f32,
}

impl RecipeMeta for AxisValue {}

#[cube]
impl<T: Float> Recipe<T> for AxisValue {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        T::cast_from(coordinates.along(self.axis)) * T::cast_from(self.scale)
    }
}

/// Walk the whole space, ringing each region of `source` through whatever residence it states,
/// and copy it into `output`. A source left [`in place`](StagePlan::in_place) is evaluated where
/// it is read; one asking for a stage is materialized into shared memory first.
#[cube]
fn materialize<E: Numeric>(
    source: &Tile<E>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
) {
    let output = output.tile(comptime!(space.clone()));
    let mut ring = Ring::unary(
        source,
        comptime!(space.clone()),
        comptime!(space.clone()),
        1usize,
    );
    // A staged slot already holds this region's window; an in-place payload is the source itself,
    // undivided, and still has to select the region out. The engine's own schedules get this from
    // `read_operand`, which a test cannot reach.
    let plan = source.stage_plan();
    let windowed = comptime!(plan.head() != Residence::InPlace);
    let walk = Walk::over(source.runtime_space());
    for region in walk {
        let staging = ring.slot_mut(0usize);
        staging.fill_streamed(source, &region);
        staging.publish();
        staging.consume(|staged| {
            let window = if comptime!(windowed) {
                staged.clone()
            } else {
                staged.at(&region)
            };
            output.at(&region).copy_from(&window);
        });
    }
}

type LinearScaled = Linear<AxisValue>;

/// `x = offset + coordinate[COL]`, built so the recipe carries genuinely runtime scalars across
/// the staging walk rather than folding to comptime.
#[cube]
fn along_col<E: Float>(#[comptime] offset: ComptimeFloat<f32>) -> AffineCoordinate<E> {
    let zero = E::cast_from(0u32.runtime());
    affine_along(
        COL,
        E::new(comptime!(offset.get())) + zero,
        E::new(1.0_f32) + zero,
    )
}

/// `stage` picks whether the recipe is evaluated at the read site or first materialized into
/// shared memory: the walk must produce the same grid either way.
#[cube(launch)]
fn product_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] stage: StagePlan,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural_resident::<ProductAxes>(
        comptime!(space.clone()),
        ProductAxes { row: ROW, col: COL },
        stage,
    );
    materialize(&source, output, space);
}

#[cube(launch)]
fn rebase_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural::<AxisValue>(
        comptime!(space.clone()),
        AxisValue {
            axis: ROW,
            scale: 2.0,
        },
    );
    // The second region starts at (2, 3), so its first logical coordinate reads row 2.
    let region = Region::trailing(comptime!(space.clone()), 1usize, 1usize);
    let source = source.at(&region);
    let mut pos = Coords::<u32>::new();
    pos.push(0u32.runtime());
    pos.push(0u32.runtime());
    let mut output = output.tile(space);
    output.init(source.procedural_value(pos));
}

#[cube(launch)]
fn constant_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural::<Constant<E>>(
        comptime!(space.clone()),
        Constant::<E> {
            value: E::new(-1.25_f32) + E::cast_from(0u32.runtime()),
        },
    );
    materialize(&source, output, space);
}

#[cube(launch)]
fn affine_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] offset: ComptimeFloat<f32>,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural::<AffineCoordinate<E>>(
        comptime!(space.clone()),
        along_col::<E>(offset),
    );
    materialize(&source, output, space);
}

#[cube(launch)]
fn linear_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] offset: ComptimeFloat<f32>,
    #[define(E)] _dtype: ElemType,
) {
    let zero = E::cast_from(0u32.runtime());
    let source = Tile::<E>::procedural::<LinearAxis<E>>(
        comptime!(space.clone()),
        linear_along(
            COL,
            E::new(comptime!(offset.get())) + zero,
            E::new(1.0_f32) + zero,
        ),
    );
    materialize(&source, output, space);
}

#[cube(launch)]
fn cubic_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] offset: ComptimeFloat<f32>,
    #[comptime] a: Ratio,
    #[define(E)] _dtype: ElemType,
) {
    let zero = E::cast_from(0u32.runtime());
    let source = Tile::<E>::procedural::<CubicAxis<E>>(
        comptime!(space.clone()),
        cubic_along(
            COL,
            E::new(comptime!(offset.get())) + zero,
            E::new(1.0_f32) + zero,
            a,
        ),
    );
    materialize(&source, output, space);
}

#[cube(launch)]
fn lanczos_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] offset: ComptimeFloat<f32>,
    #[comptime] lobes: u8,
    #[define(E)] _dtype: ElemType,
) {
    let zero = E::cast_from(0u32.runtime());
    let source = Tile::<E>::procedural::<LanczosAxis<E>>(
        comptime!(space.clone()),
        lanczos_along(
            COL,
            E::new(comptime!(offset.get())) + zero,
            E::new(1.0_f32) + zero,
            lobes,
        ),
    );
    materialize(&source, output, space);
}

/// A filter over a recipe that is not an [`AffineCoordinate`], which is what the filters being
/// generic over their inner recipe buys.
#[cube(launch)]
fn linear_over_axis_value_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural::<LinearScaled>(
        comptime!(space.clone()),
        LinearScaled {
            coordinate: AxisValue {
                axis: ROW,
                scale: 0.5,
            },
        },
    );
    materialize(&source, output, space);
}

/// A procedural tile over an integer element type: [`Recipe`] is defined over `Numeric`, so
/// nothing about a procedural source requires floats.
#[cube(launch)]
fn integer_kernel<E: Int>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural::<Constant<E>>(
        comptime!(space.clone()),
        Constant::<E> {
            value: E::new(7) + E::cast_from(0u32.runtime()),
        },
    );
    materialize(&source, output, space);
}

/// A direct procedural read must use the masked view path on trailing partial tiles.
#[cube(launch)]
fn direct_copy_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural::<Constant<E>>(
        comptime!(space.clone()),
        Constant::<E> {
            value: E::new(1.0_f32) + E::cast_from(0u32.runtime()),
        },
    );
    let mut output = output.tile(space);
    output.copy_from(&source);
}

/// Descend once before a direct read: the procedural bound remains in the parent coordinates.
#[cube(launch)]
fn divided_direct_copy_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let source = Tile::<E>::procedural::<Constant<E>>(
        comptime!(space.clone()),
        Constant::<E> {
            value: E::new(1.0_f32) + E::cast_from(0u32.runtime()),
        },
    );
    let region = Region::trailing(comptime!(space.clone()), 0usize, 0usize);
    let source = source.at(&region);
    let output = output.tile(comptime!(space.clone()));
    let mut output = output.at(&region);
    output.copy_from(&source);
}

struct Harness {
    client: ComputeClient<TestRuntime>,
    dtype: ElemType,
    space: Space,
}

impl Harness {
    fn new() -> Self {
        Self {
            client: <TestRuntime as Runtime>::client(&Default::default()),
            dtype: f32::elem_type_native(),
            space: Tiling::new()
                .extents(&[(ROW, ROWS), (COL, COLS)])
                .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
                    level
                        .axis(ROW, Cut::sequential(2))
                        .axis(COL, Cut::sequential(3))
                })
                .build(),
        }
    }

    fn output(&self) -> TensorHandle<TestRuntime> {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .zeros()
            .generate_without_host_data()
    }

    fn read(&self, output: TensorHandle<TestRuntime>) -> HostData {
        HostData::from_tensor_handle(&self.client, output, HostDataType::F32)
    }
}

/// The output launch argument. Each kernel gets its own opaque element type, so this cannot be a
/// function.
macro_rules! output_arg {
    ($output:expr) => {
        TileArgLaunch::new(
            $output.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[ROW, COL]),
        )
    };
}

/// Assert every cell equals `expected(row, col)`, tolerating float rounding.
fn assert_grid(got: &HostData, expected: impl Fn(usize, usize) -> f32) {
    for row in 0..ROWS {
        for col in 0..COLS {
            let want = expected(row, col);
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-5,
                "at ({row}, {col}): got {have}, want {want}"
            );
        }
    }
}

/// A finite comptime offset for [`along_col`].
fn offset(value: f32) -> ComptimeFloat<f32> {
    ComptimeFloat::new(value).unwrap()
}

/// `sin(pi t) / (pi t)`, the definition [`Lanczos`] is derived from.
fn sinc(t: f32) -> f32 {
    if t.abs() < 1e-7 {
        1.0
    } else {
        (PI * t).sin() / (PI * t)
    }
}

/// Walk the product recipe with the source staged as `stage` says; the grid is the same either
/// way.
fn check_product(stage: StagePlan) {
    let h = Harness::new();
    let output = h.output();
    product_kernel::launch::<TestRuntime>(
        &h.client,
        h.space.cube_count(),
        h.space.cube_dim(&h.client),
        output_arg!(output),
        h.space.clone(),
        stage,
        h.dtype,
    );
    assert_grid(&h.read(output), |row, col| (row * col) as f32);
}

#[test]
fn user_recipe_evaluates_in_place() {
    check_product(StagePlan::in_place());
}

#[test]
fn user_recipe_materializes_through_a_staged_walk() {
    check_product(StagePlan::new(&[Residence::Smem], StageStorage::Strided, 0));
}

#[test]
fn selecting_a_region_rebases_the_recipe_origin() {
    let h = Harness::new();
    let output = h.output();
    rebase_kernel::launch::<TestRuntime>(
        &h.client,
        h.space.cube_count(),
        h.space.cube_dim(&h.client),
        output_arg!(output),
        h.space.clone(),
        h.dtype,
    );
    assert_grid(&h.read(output), |_, _| 4.0);
}

#[test]
fn constant_evaluates_its_value_everywhere() {
    let h = Harness::new();
    let output = h.output();
    constant_kernel::launch::<TestRuntime>(
        &h.client,
        h.space.cube_count(),
        h.space.cube_dim(&h.client),
        output_arg!(output),
        h.space.clone(),
        h.dtype,
    );
    assert_grid(&h.read(output), |_, _| -1.25);
}

#[test]
fn affine_coordinates_evaluate_absolute_positions() {
    let h = Harness::new();
    let output = h.output();
    affine_kernel::launch::<TestRuntime>(
        &h.client,
        h.space.cube_count(),
        h.space.cube_dim(&h.client),
        output_arg!(output),
        h.space.clone(),
        offset(-2.5),
        h.dtype,
    );
    // Constant down each column: the recipe reads COL only, and the value must hold for the lower
    // regions of the walk, whose origin the rebasing has moved.
    assert_grid(&h.read(output), |_, col| -2.5 + col as f32);
}

#[test]
fn linear_is_a_triangle_with_unit_support() {
    let h = Harness::new();
    let output = h.output();
    linear_kernel::launch::<TestRuntime>(
        &h.client,
        h.space.cube_count(),
        h.space.cube_dim(&h.client),
        output_arg!(output),
        h.space.clone(),
        offset(-2.5),
        h.dtype,
    );
    // x runs -2.5 ..= 2.5, so both the support and the cutoff are sampled.
    assert_grid(&h.read(output), |_, col| {
        let x = (-2.5 + col as f32).abs();
        if x < 1.0 { 1.0 - x } else { 0.0 }
    });
}

#[test]
fn a_procedural_tile_works_over_an_integer_element_type() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = i32::elem_type_native();
    let space = Harness::new().space;
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    integer_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        output_arg!(output),
        space.clone(),
        dtype,
    );
    let got = HostData::from_tensor_handle(&client, output, HostDataType::I32);
    for row in 0..ROWS {
        for col in 0..COLS {
            assert_eq!(got.get_i32(&[row, col]), 7);
        }
    }
}

#[test]
fn a_filter_wraps_any_recipe_not_only_affine_coordinates() {
    let h = Harness::new();
    let output = h.output();
    linear_over_axis_value_kernel::launch::<TestRuntime>(
        &h.client,
        h.space.cube_count(),
        h.space.cube_dim(&h.client),
        output_arg!(output),
        h.space.clone(),
        h.dtype,
    );
    // x = row / 2, so the triangle falls to zero at row 2 and stays there.
    assert_grid(&h.read(output), |row, _| {
        let x = row as f32 / 2.0;
        if x < 1.0 { 1.0 - x } else { 0.0 }
    });
}

#[test]
fn cubic_matches_the_keys_convolution() {
    // Both `a` values the presets pick: catmull_rom is -1/2, sharp is -3/4.
    for ratio in [Ratio::new(-1, 2), Ratio::new(-3, 4)] {
        let a = ratio.as_f32();
        let h = Harness::new();
        let output = h.output();
        cubic_kernel::launch::<TestRuntime>(
            &h.client,
            h.space.cube_count(),
            h.space.cube_dim(&h.client),
            output_arg!(output),
            h.space.clone(),
            offset(-2.5),
            ratio,
            h.dtype,
        );
        // x runs -2.5 ..= 2.5, covering all three pieces of the kernel.
        assert_grid(&h.read(output), |_, col| {
            let x = (-2.5 + col as f32).abs();
            if x <= 1.0 {
                (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0
            } else if x <= 2.0 {
                a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a
            } else {
                0.0
            }
        });
    }
}

#[test]
fn lanczos_matches_the_windowed_sinc() {
    // The first case samples half-integers and the |x| = lobes cutoff; the second lands a sample
    // on the x = 0 singularity and on the sinc zeros.
    for (start, lobes) in [(-2.5f32, 2u8), (-2.0f32, 3u8)] {
        let h = Harness::new();
        let output = h.output();
        lanczos_kernel::launch::<TestRuntime>(
            &h.client,
            h.space.cube_count(),
            h.space.cube_dim(&h.client),
            output_arg!(output),
            h.space.clone(),
            offset(start),
            lobes,
            h.dtype,
        );
        assert_grid(&h.read(output), |_, col| {
            let x = start + col as f32;
            let lobes = lobes as f32;
            if x.abs() >= lobes {
                0.0
            } else {
                sinc(x) * sinc(x / lobes)
            }
        });
    }
}

#[test]
fn direct_copy_masks_the_trailing_partial_tile() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let space = Tiling::new()
        .extents(&[(ROW, ROWS), (COL, COLS)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(4))
        })
        .build();
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    direct_copy_kernel::launch::<TestRuntime>(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        output_arg!(output),
        space,
        dtype,
    );

    assert_grid(
        &HostData::from_tensor_handle(&client, output, HostDataType::F32),
        |_, _| 1.0,
    );
}

#[test]
fn divided_direct_copy_preserves_the_parent_bound() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let concrete = Tiling::new()
        .extents(&[(ROW, ROWS), (COL, COLS)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |level| {
            level
                .axis(ROW, Cut::sequential(2))
                .axis(COL, Cut::sequential(4))
        })
        .build();
    let space = concrete.clone().with_dynamic(&[ROW]);
    let output = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    divided_direct_copy_kernel::launch::<TestRuntime>(
        &client,
        concrete.cube_count(),
        concrete.cube_dim(&client),
        output_arg!(output),
        space,
        dtype,
    );

    assert_grid(
        &HostData::from_tensor_handle(&client, output, HostDataType::F32),
        |row, col| if row < 2 && col < 4 { 1.0 } else { 0.0 },
    );
}

#[test]
fn recipe_facts_compose_through_nesting() {
    assert_eq!(RecipeFacts::of::<Zeros>().halo, 0);
    assert_eq!(RecipeFacts::of::<Ones>().halo, 0);
    assert_eq!(RecipeFacts::of::<Constant<f32>>().halo, 0);
    assert_eq!(RecipeFacts::of::<AffineCoordinate<f32>>().halo, 0);
    assert_eq!(RecipeFacts::of::<LinearAxis<f32>>().halo, 1);
    assert_eq!(RecipeFacts::of::<CubicAxis<f32>>().halo, 2);
    assert_eq!(RecipeFacts::of::<LanczosAxis<f32>>().halo, 3);
    assert_eq!(
        RecipeFacts::of::<Lanczos<Cubic<AffineCoordinate<f32>>>>().halo,
        5
    );
    assert_eq!(
        RecipeFacts::of::<Product<CubicAxis<f32>, LanczosAxis<f32>>>().halo,
        3
    );
}
