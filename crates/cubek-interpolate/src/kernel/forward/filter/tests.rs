//! The resampling filters, each evaluated as a procedural tile against its closed form: a
//! triangle, the Keys cubic at both presets' `a`, and the Lanczos windowed sinc, plus one filter
//! over a recipe that is not an affine coordinate.

use core::f32::consts::PI;

use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::shape};
use cubecl_common::{ComptimeFloat, Ratio};
use cubek_test_utils::{HostData, HostDataType, TestInput};
use cubek_tile::launch::Grid;
use cubek_tile::procedural::Procedural;
use cubek_tile::procedural::Recipe;
use cubek_tile::procedural::RecipeCoords;
use cubek_tile::procedural::RecipeExpand;
use cubek_tile::*;

use super::{CubicAxis, LanczosAxis, Linear, LinearAxis, cubic_along, lanczos_along, linear_along};

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const ROWS: usize = 4;
const COLS: usize = 6;

#[derive(CubeType, Clone)]
struct AxisValue {
    #[cube(comptime)]
    axis: Axis,
    scale: f32,
}

#[cube]
impl<T: Float> Recipe<T> for AxisValue {
    fn evaluate(&self, coordinates: &RecipeCoords) -> T {
        T::cast_from(coordinates.along(self.axis)) * T::cast_from(self.scale)
    }
}

/// Walk the whole space and copy each region of `source`, evaluated where it is read, into
/// `output`.
#[cube]
fn materialize<E: Numeric>(
    source: &Tile<E>,
    output: &TileArg<'_, E, Const<1>>,
    space: &Partitioning,
    #[comptime] level: Level,
) {
    let output = output.tile(comptime!(space.clone()));
    for region in source.over(&level) {
        let mut output_region = output.at(&region);
        output_region.copy_from(&source.at(&region));
    }
}

/// Forces a scalar value into a runtime GPU variable by adding a runtime zero,
/// preventing the expander from folding it away at compile time.
#[cube]
fn runtime_scalar<E: Numeric>(value: E) -> E {
    value + E::cast_from(0u32.runtime())
}

struct Harness {
    client: Client,
    dtype: ElemType,
    launcher: Launcher,
}

impl Harness {
    fn new() -> Self {
        Self {
            client: cubecl::test_device().client(),
            dtype: f32::elem_type_native(),
            launcher: Launcher::new(
                &cubecl::test_device().client(),
                Partitioning::new(
                    Space::new(&[(ROW, ROWS), (COL, COLS)]),
                    Levels::leaf(&[(ROW, 2), (COL, 3)])
                        .walk_every(&[ROW, COL])
                        .build(),
                ),
                &Space::new(&[(ROW, ROWS), (COL, COLS)]),
                Grid::FromLevels,
            ),
        }
    }

    fn output(&self) -> TensorHandle {
        TestInput::builder(self.client.clone(), shape![ROWS, COLS])
            .dtype(self.dtype)
            .zeros()
            .generate_without_host_data()
    }

    fn read(&self, output: TensorHandle) -> HostData {
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

/// A finite comptime offset for a filter's coordinate.
fn offset(value: f32) -> ComptimeFloat<f32> {
    ComptimeFloat::new(value).unwrap()
}

type LinearScaled = Linear<AxisValue>;

#[cube(launch)]
fn linear_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] offset: ComptimeFloat<f32>,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::<E>::new::<LinearAxis<E>>(
        comptime!(space.space().clone()),
        linear_along(
            COL,
            runtime_scalar::<E>(E::new(comptime!(offset.get()))),
            runtime_scalar::<E>(E::new(1.0_f32)),
        ),
    )
    .tile();
    materialize(&source, output, &space, level.clone());
}

#[cube(launch)]
fn cubic_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] offset: ComptimeFloat<f32>,
    #[comptime] a: Ratio,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::<E>::new::<CubicAxis<E>>(
        comptime!(space.space().clone()),
        cubic_along(
            COL,
            runtime_scalar::<E>(E::new(comptime!(offset.get()))),
            runtime_scalar::<E>(E::new(1.0_f32)),
            a,
        ),
    )
    .tile();
    materialize(&source, output, &space, level.clone());
}

#[cube(launch)]
fn lanczos_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] offset: ComptimeFloat<f32>,
    #[comptime] lobes: u8,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::<E>::new::<LanczosAxis<E>>(
        comptime!(space.space().clone()),
        lanczos_along(
            COL,
            runtime_scalar::<E>(E::new(comptime!(offset.get()))),
            runtime_scalar::<E>(E::new(1.0_f32)),
            lobes,
        ),
    )
    .tile();
    materialize(&source, output, &space, level.clone());
}

/// A filter over a recipe that is not an [`AffineCoordinate`], which is what the filters being
/// generic over their inner recipe buys.
#[cube(launch)]
fn linear_over_axis_value_kernel<E: Float>(
    output: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let source = Procedural::<E>::new::<LinearScaled>(
        comptime!(space.space().clone()),
        LinearScaled {
            coordinate: AxisValue {
                axis: ROW,
                scale: 0.5,
            },
        },
    )
    .tile();
    materialize(&source, output, &space, level.clone());
}

/// `sin(pi t) / (pi t)`, the definition [`Lanczos`] is derived from.
fn sinc(t: f32) -> f32 {
    if t.abs() < 1e-7 {
        1.0
    } else {
        (PI * t).sin() / (PI * t)
    }
}

#[test]
fn linear_is_a_triangle_with_unit_support() {
    let h = Harness::new();
    let output = h.output();
    linear_kernel::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
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
fn a_filter_wraps_any_recipe_not_only_affine_coordinates() {
    let h = Harness::new();
    let output = h.output();
    linear_over_axis_value_kernel::launch(
        &h.client,
        h.launcher.cube_count(),
        h.launcher.cube_dim(),
        output_arg!(output),
        h.launcher.partitioning_arg(),
        h.launcher.partitioning().level(0),
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
        cubic_kernel::launch(
            &h.client,
            h.launcher.cube_count(),
            h.launcher.cube_dim(),
            output_arg!(output),
            h.launcher.partitioning_arg(),
            h.launcher.partitioning().level(0),
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
        lanczos_kernel::launch(
            &h.client,
            h.launcher.cube_count(),
            h.launcher.cube_dim(),
            output_arg!(output),
            h.launcher.partitioning_arg(),
            h.launcher.partitioning().level(0),
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
