//! Spike: can `#[cube]` carry an `Operand` trait that both `Tile<T>` and a scales-decorated
//! pair implement, without pushing anything comptime to runtime?

use cubecl::prelude::*;
use cubek_tile::*;

/// What the contraction asks of an operand. `W` rides the trait rather than each method, so no
/// method is itself generic.
#[cube]
trait Operand<T: Numeric, W: Size>: CubeType {
    fn vector_size(&self) -> comptime_type!(usize);
    fn op_space(&self) -> comptime_type!(Space);
    fn at_region(&self, region: &Region) -> Self;
    fn matrix_of(&self, #[comptime] axes: MatrixAxes, i: usize) -> MatrixView<'_, Vector<T, W>>;
}

#[cube]
impl<T: Numeric, W: Size> Operand<T, W> for Tile<T> {
    fn vector_size(&self) -> comptime_type!(usize) {
        self.vector_size()
    }
    fn op_space(&self) -> comptime_type!(Space) {
        comptime!(self.space.clone())
    }
    fn at_region(&self, region: &Region) -> Tile<T> {
        self.at(region)
    }
    fn matrix_of(&self, #[comptime] axes: MatrixAxes, i: usize) -> MatrixView<'_, Vector<T, W>> {
        self.matrix_packed::<W>(axes, i)
    }
}

/// The values with their scales: two owned tiles, no boxing, no recursion.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
struct Scaled<T: Numeric, S: Numeric> {
    values: Tile<T>,
    scales: Tile<S>,
    #[cube(comptime)]
    scale_axes: MatrixAxes,
}

#[cube]
impl<T: Numeric, W: Size, S: Numeric> Operand<T, W> for Scaled<T, S> {
    fn vector_size(&self) -> comptime_type!(usize) {
        self.values.vector_size()
    }
    fn op_space(&self) -> comptime_type!(Space) {
        comptime!(self.values.space.clone())
    }
    fn at_region(&self, region: &Region) -> Scaled<T, S> {
        Scaled::<T, S> {
            values: self.values.at(region),
            scales: self.scales.at(region),
            scale_axes: comptime!(self.scale_axes),
        }
    }
    fn matrix_of(&self, #[comptime] axes: MatrixAxes, i: usize) -> MatrixView<'_, Vector<T, W>> {
        self.values.matrix_scaled::<W, S, Const<1>>(
            axes,
            &self.scales,
            comptime!(self.scale_axes),
            i,
        )
    }
}

/// The shape `mma_leaf` would take: generic over the operand, comptime result used where a
/// comptime result must be.
#[cube]
fn leaf<T: Numeric, W: Size, L: Operand<T, W>>(
    operand: &L,
    #[comptime] axes: MatrixAxes,
    out: &mut Tensor<Vector<T, W>>,
) {
    let space = operand.op_space();
    let cols = comptime!(space.extent(Axis(1)));
    // `at_region` returns `Self`: the pair descends both tiles in lockstep, like the walk does.
    let windowed = operand.at_region(&Region::trailing(space, 0usize, 0usize));
    let mat = windowed.matrix_of(axes, 0);
    let width = windowed.vector_size();
    comptime!(assert!(width == 1));
    #[unroll(comptime!(cols == 4))]
    for i in 0..comptime!(cols) {
        out[i] = mat.read((0, i as u32));
    }
}

/// The same body written against a concrete `Tile`, for the A/B source diff.
#[cube]
fn leaf_concrete<T: Numeric, W: Size>(
    tile: &Tile<T>,
    #[comptime] axes: MatrixAxes,
    out: &mut Tensor<Vector<T, W>>,
) {
    let cols = comptime!(tile.space.extent(Axis(1)));
    let windowed = tile.at(&Region::trailing(
        comptime!(tile.space.clone()),
        0usize,
        0usize,
    ));
    let mat = windowed.matrix_packed::<W>(axes, 0);
    let width = windowed.vector_size();
    comptime!(assert!(width == 1));
    #[unroll(comptime!(cols == 4))]
    for i in 0..comptime!(cols) {
        out[i] = mat.read((0, i as u32));
    }
}

#[cube(launch)]
fn concrete_kernel<T: Numeric>(
    a: &TileArg<'_, T, Const<1>>,
    out: &mut Tensor<Vector<T, Const<1>>>,
    #[comptime] space: Space,
    #[define(T)] _dtype: ElemType,
) {
    let tile = a.tile(comptime!(space.clone()));
    leaf_concrete::<T, Const<1>>(&tile, comptime!(MatrixAxes::trailing_pair(&space)), out);
}

#[cube(launch)]
fn trait_kernel<T: Numeric>(
    a: &TileArg<'_, T, Const<1>>,
    out: &mut Tensor<Vector<T, Const<1>>>,
    #[comptime] space: Space,
    #[define(T)] _dtype: ElemType,
) {
    let tile = a.tile(comptime!(space.clone()));
    leaf::<T, Const<1>, Tile<T>>(&tile, comptime!(MatrixAxes::trailing_pair(&space)), out);
}

#[test]
fn the_trait_and_the_concrete_call_compile_the_same_kernel() {
    use cubecl::{Runtime, TestRuntime, zspace::shape};
    use cubek_test_utils::TestInput;

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let (rows, cols) = (4, 4);
    let space = Tiling::new()
        .extents(&[(Axis(0), rows), (Axis(1), cols)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(Axis(0), Cut::sequential(rows))
                .axis(Axis(1), Cut::sequential(cols))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    let (a, _) = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .arange()
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    concrete_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            a.binding().into_tensor_arg(),
            TileSpec::direct(&[Axis(0), Axis(1)]),
        ),
        out.binding().into_tensor_arg(),
        space.clone(),
        dtype,
    );
    cubecl::future::block_on(client.sync());

    let (a, _) = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .arange()
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    trait_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            a.binding().into_tensor_arg(),
            TileSpec::direct(&[Axis(0), Axis(1)]),
        ),
        out.binding().into_tensor_arg(),
        space,
        dtype,
    );
    cubecl::future::block_on(client.sync());
}

/// The scaled operand through the *same* generic leaf: one body, two impls.
#[cube(launch)]
fn scaled_trait_kernel<T: Numeric, S: Numeric>(
    a: &TileArg<'_, T, Const<1>>,
    s: &TileArg<'_, S, Const<1>>,
    out: &mut Tensor<Vector<T, Const<1>>>,
    #[comptime] space: Space,
    #[define(T, S)] _dtypes: [ElemType; 2],
) {
    let values = a.tile(comptime!(space.clone()));
    let scales = s.tile(comptime!(space.clone()));
    let axes = comptime!(MatrixAxes::trailing_pair(&space));
    let operand = Scaled::<T, S> {
        values,
        scales,
        scale_axes: axes,
    };
    leaf::<T, Const<1>, Scaled<T, S>>(&operand, axes, out);
}

#[test]
fn the_same_generic_leaf_reads_a_scaled_operand() {
    use cubecl::{Runtime, TestRuntime, zspace::shape};
    use cubek_test_utils::{HostData, HostDataType, TestInput};

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::elem_type_native();
    let (rows, cols) = (4, 4);
    let space = Tiling::new()
        .extents(&[(Axis(0), rows), (Axis(1), cols)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(Axis(0), Cut::sequential(rows))
                .axis(Axis(1), Cut::sequential(cols))
        })
        .build()
        .with_instruction(Instruction::registers(16));

    let a: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    // One scale per column, broadcast over the rows: the operand omits Axis(0).
    let s: Vec<f32> = vec![2.0, 3.0, 4.0, 5.0];

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    scaled_trait_kernel::launch::<TestRuntime>(
        &client,
        CubeCount::new_single(),
        CubeDim::new_single(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            TileSpec::direct(&[Axis(0), Axis(1)]),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            TileSpec::direct(&[Axis(0), Axis(1)]),
        ),
        out.clone().binding().into_tensor_arg(),
        space,
        [dtype, dtype],
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for n in 0..cols {
        let want = a[n] * s[n];
        let have = got.get_f32(&[n]);
        assert!(
            (have - want).abs() < 1e-6,
            "at {n}: got {have}, want {want}"
        );
    }
}
