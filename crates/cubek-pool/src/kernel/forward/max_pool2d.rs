use super::{
    super::shape_divmod,
    pool2d::{
        Pool2dDirectArgsLaunch, Pool2dDirectStrategy, Pool2dDirectStrategyFamily, Position,
        pool2d_direct, view4d,
    },
};
use crate::definition::{MaxPoolOptions, PoolError};
use cubecl::{
    CubeDim, Runtime, calculate_cube_count_elemwise, num_traits::Zero, prelude::TensorBinding,
    prelude::*, std::tensor::View, tensor_vector_size_parallel,
};

struct MaxPoolStrategy;
struct MaxPoolWithIndicesStrategy;

impl Pool2dDirectStrategyFamily for MaxPoolStrategy {
    type Indices<N: Size> = ();
    type Config = ();
    type Pool2d<T: Numeric, N: Size> = Self;
}

impl Pool2dDirectStrategyFamily for MaxPoolWithIndicesStrategy {
    type Indices<N: Size> = View<Vector<i32, N>, Position, ReadWrite>;
    type Config = ();
    type Pool2d<T: Numeric, N: Size> = Self;
}

#[cube]
impl<T: Numeric, N: Size> Pool2dDirectStrategy<T, N> for MaxPoolStrategy {
    type Accumulator = Vector<T, N>;
    type Config = ();
    type Indices = ();

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        Vector::new(T::min_value())
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        _index: VectorSize,
        result: Vector<T, N>,
    ) {
        *accumulator = max(*accumulator, result);
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _ih: u32,
        _iw: u32,
    ) {
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        _output_indices: &mut (),
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator;
    }
}

#[cube]
impl<T: Numeric, N: Size> Pool2dDirectStrategy<T, N> for MaxPoolWithIndicesStrategy {
    type Accumulator = (Vector<T, N>, Vector<i32, N>);
    type Config = ();
    type Indices = View<Vector<i32, N>, Position, ReadWrite>;

    fn initialize(#[comptime] _config: &Self::Config) -> Self::Accumulator {
        let val = Vector::new(T::min_value());
        let idx = Vector::zero();
        (val, idx)
    }

    fn accumulate(
        #[comptime] _config: &Self::Config,
        accumulator: &mut Self::Accumulator,
        index: usize,
        result: Vector<T, N>,
    ) {
        let indices = Vector::cast_from(index);
        accumulator.1 = select_many(result.greater_than(accumulator.0), indices, accumulator.1);
        accumulator.0 = max(result, accumulator.0);
    }

    fn count_position(
        #[comptime] _config: &Self::Config,
        _accumulator: &mut Self::Accumulator,
        _ih: u32,
        _iw: u32,
    ) {
    }

    fn store(
        #[comptime] _config: &Self::Config,
        position: Position,
        output: &mut View<Vector<T, N>, Position, ReadWrite>,
        output_indices: &mut View<Vector<i32, N>, Position, ReadWrite>,
        accumulator: Self::Accumulator,
    ) {
        output[position] = accumulator.0;
        output_indices[position] = accumulator.1;
    }
}

pub(crate) fn max_pool2d_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: MaxPoolOptions<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    pool2d_direct::launch::<MaxPoolStrategy, R>(
        client,
        cube_count,
        cube_dim,
        address_type,
        vector_size,
        input.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        (),
        shape_divmod(&output),
        working_units,
        Pool2dDirectArgsLaunch::new(
            options.window.stride[0] as u32,
            options.window.stride[1] as u32,
            options.dilation[0] as u32,
            options.dilation[1] as u32,
            options.window.padding[0] as u32,
            options.window.padding[1] as u32,
        ),
        (
            options.window.kernel_size[0] as u32,
            options.window.kernel_size[1] as u32,
        ),
        (),
        dtype,
    );

    Ok(())
}

pub(crate) fn max_pool2d_with_indices_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    indices: TensorBinding<R>,
    options: MaxPoolOptions<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()))
        .max(indices.required_address_type(dtype.size()));

    pool2d_direct::launch::<MaxPoolWithIndicesStrategy, R>(
        client,
        cube_count,
        cube_dim,
        address_type,
        vector_size,
        input.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        view4d(indices.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        Pool2dDirectArgsLaunch::new(
            options.window.stride[0] as u32,
            options.window.stride[1] as u32,
            options.dilation[0] as u32,
            options.dilation[1] as u32,
            options.window.padding[0] as u32,
            options.window.padding[1] as u32,
        ),
        (
            options.window.kernel_size[0] as u32,
            options.window.kernel_size[1] as u32,
        ),
        (),
        dtype,
    );

    Ok(())
}
