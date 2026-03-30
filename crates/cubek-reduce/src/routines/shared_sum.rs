use cubecl::{
    features::AtomicUsage,
    std::tensor::layout::linear::{
        LinearView, LinearViewLaunch, LinearViewLayout, LinearViewLayoutLaunch,
    },
};
use cubecl::{ir::ElemType, std::tensor::layout::linear::linear_view};
use cubecl::{prelude::*, tensor_vector_size_parallel};

use crate::ReduceError;

/// Sum all the elements of the input tensor distributed over `cube_count` cubes.
///
/// This is an optimized version for summing large tensors using multiple cubes.
/// For summing a single axis, the regular reduce entry point is preferred.
///
/// Return an error if atomic addition is not supported for the type `N`.
///
/// # Important
///
/// This doesn't set the value of output to 0 before computing the sums.
/// It is the responsibility of the caller to ensure that output is set to
/// the proper value. Basically, the behavior of this kernel is akin to the AddAssign operator
/// as it update the output instead of overwriting it.
///
/// # Example
///
/// This examples show how to sum all the elements of a small `2 x 2` matrix.
/// For more details, see the CubeCL documentation.
///
/// ```ignore
/// let client = /* ... */;
/// let size_f32 = std::mem::size_of::<f32>();
///
/// // Create input and output handles.
/// let input_handle = client.create(f32::as_bytes(&[0, 1, 2, 3]));
/// let output_handle = client.empty(size_of::<F>());
/// let input = unsafe {
///     TensorBinding::from_raw_parts(
///         &input_handle,
///         &[2, 1],
///         &[2, 2],
///         size_f32,
///     )
/// };
/// let output = unsafe {
///     TensorBinding::from_raw_parts(&output_handle, &[1], &[1], size_of::<F>())
/// };
///
/// // Here `R` is a `cubecl::Runtime`.
/// let result = shared_sum::<R, f32>(&client, input, output, cube_count);
///
/// if result.is_ok() {
///        let binding = output_handle.binding();
///        let bytes = client.read_one(binding);
///        let output_values = f32::from_bytes(&bytes);
///        println!("Output = {:?}", output_values); // Should print [6].
/// }
/// ```
pub fn shared_sum<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    cube_count: u32,
    input_elem: ElemType,
) -> Result<(), ReduceError> {
    // Check that the client supports atomic addition.
    if !client
        .properties()
        .atomic_type_usage(Type::new(StorageType::Atomic(input_elem)))
        .contains(AtomicUsage::Add)
    {
        return Err(ReduceError::MissingAtomicAdd(input_elem.into()));
    }

    let input_len = input.shape.iter().product::<usize>();
    let contiguous_buffer = input_len * input_elem.size() == input.handle.size_in_used() as usize;

    // Compute the optimal vector size.
    let vector_size = if contiguous_buffer {
        client
            .io_optimized_vector_sizes(input_elem.size())
            .filter(|vector_size| input_len.is_multiple_of(*vector_size))
            .max()
            .unwrap_or(1)
    } else {
        tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(input_elem.size()),
            &input.shape,
            &input.strides,
            input.shape.len() - 1,
        )
    };

    let address_type = input
        .required_address_type(input_elem.size())
        .max(output.required_address_type(input_elem.size()));

    // Sum is commutative so we don't care about order, but need to care if there are holes since
    // they're not guaranteed to contain `0`.
    let input_view = if contiguous_buffer {
        let layout = LinearViewLayoutLaunch::new();
        let buffer = unsafe { ArrayArg::from_raw_parts_binding(input.handle, input_len) };
        LinearViewLaunch::new_array::<LinearViewLayout>(buffer, layout)
    } else {
        linear_view(input)
    };

    // Compute extra parameters.
    let cube_dim = CubeDim::new_2d(32, 8); // NOTE: If you change that, keep the unit count a power of 2.
    let num_units = cube_count * cube_dim.num_elems();
    let num_vectors_per_unit = input_len.div_ceil(num_units as usize * vector_size);
    let cube_count = CubeCount::new_1d(cube_count);

    // Launch kernel
    unsafe {
        shared_sum_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            input_view,
            output.into_tensor_arg(),
            cube_dim.num_elems() as usize,
            num_vectors_per_unit,
            input_elem,
        )
    };

    Ok(())
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn shared_sum_kernel<T: Numeric, N: Size>(
    input: &LinearView<Vector<T, N>>,
    output: &mut Tensor<Atomic<T>>,
    #[comptime] shared_memory_size: usize,
    #[comptime] num_vectors_per_unit: usize,
    #[define(T)] _dtype: ElemType,
) {
    let mut shared_memory = SharedMemory::new(shared_memory_size);
    shared_memory[UNIT_POS as usize] = Vector::empty().fill(T::from_int(0));

    // Each unit reduce `num_vectors_per_unit` vectors.
    let start = ABSOLUTE_POS * num_vectors_per_unit;
    let end = start + num_vectors_per_unit;

    // Prevent out-of-bound access
    let start = select(start < input.shape(), start, input.shape());
    let end = select(end < input.shape(), end, input.shape());

    // Each unit sum its vectors.
    for k in start..end {
        shared_memory[UNIT_POS as usize] += input[k];
    }

    // Sum all vectors within the shared_memory to a single vector.
    let vector = sum_shared_memory(&mut shared_memory);

    // Sum all the elements within the vector.
    let sum = RuntimeCell::<T>::new(T::from_int(0));
    #[unroll]
    for k in 0..N::value() {
        let update = vector[k] + sum.read();
        sum.store(update);
    }

    // Add the sum for the current cube to the output.
    if UNIT_POS == 0 {
        output[0].fetch_add(sum.consume());
    }
}

// This is a simplified version of [tree_reduce].
// See the documentation there for details.
// Here we assume that `CUBE_DIM` is always a power of two.
#[cube]
fn sum_shared_memory<T: Numeric, N: Size>(
    accumulator: &mut SharedMemory<Vector<T, N>>,
) -> Vector<T, N> {
    sync_cube();
    let mut num_active_units = CUBE_DIM;
    let mut jump = 1;
    while num_active_units > 1 {
        num_active_units /= 2;
        let destination = jump * 2 * UNIT_POS;
        let origin = jump * (2 * UNIT_POS + 1);
        if UNIT_POS < num_active_units {
            let element = accumulator[origin as usize];
            accumulator[destination as usize] += element;
        }
        jump *= 2;
        sync_cube();
    }
    accumulator[0]
}
