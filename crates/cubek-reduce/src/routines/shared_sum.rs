use cubecl::{
    features::TypeUsage,
    std::tensor::layout::{
        linear::{LinearLayout, LinearLayoutArgs, LinearView, LinearViewLaunch},
        plain::PlainLayoutLaunch,
    },
};
use cubecl::{ir::ElemType, std::tensor::layout::linear::linear_view};
use cubecl::{prelude::*, tensor_line_size_parallel};

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
///     TensorHandleRef::from_raw_parts(
///         &input_handle,
///         &[2, 1],
///         &[2, 2],
///         size_f32,
///     )
/// };
/// let output = unsafe {
///     TensorHandleRef::from_raw_parts(&output_handle, &[1], &[1], size_of::<F>())
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
    input: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    cube_count: u32,
    input_elem: ElemType,
) -> Result<(), ReduceError> {
    // Check that the client supports atomic addition.
    if !client
        .properties()
        .type_usage(StorageType::Atomic(input_elem))
        .contains(TypeUsage::AtomicAdd)
    {
        return Err(ReduceError::MissingAtomicAdd(input_elem.into()));
    }

    let input_len = input.shape.iter().product::<usize>();
    let contiguous_buffer = input_len * input.elem_size == input.handle.size() as usize;

    // Compute the optimal line size.
    let line_size = if contiguous_buffer {
        client
            .io_optimized_line_sizes_unchecked(input.elem_size)
            .filter(|line_size| input_len % *line_size == 0)
            .max()
            .unwrap_or(1)
    } else {
        tensor_line_size_parallel(
            client.io_optimized_line_sizes_unchecked(input.elem_size),
            input.shape,
            input.strides,
            input.shape.len() - 1,
        )
    };

    // Sum is commutative so we don't care about order, but need to care if there are holes since
    // they're not guaranteed to contain `0`.
    let input_view = if contiguous_buffer {
        let layout = LinearLayoutArgs::Plain(PlainLayoutLaunch::new(ScalarArg::new(
            input_len / line_size,
        )));
        let buffer = unsafe {
            ArrayArg::from_raw_parts_and_size(input.handle, input_len, line_size, input.elem_size)
        };
        LinearViewLaunch::new::<LinearLayout>(buffer, layout)
    } else {
        linear_view(client, &input, line_size)
    };

    // Compute extra parameters.
    let cube_dim = CubeDim::new_2d(32, 8); // NOTE: If you change that, keep the unit count a power of 2.
    let num_units = cube_count * cube_dim.num_elems();
    let num_lines_per_unit = input_len.div_ceil(num_units as usize * line_size);
    let cube_count = CubeCount::new_1d(cube_count);

    // Launch kernel
    let result = unsafe {
        shared_sum_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            input_view,
            output.as_tensor_arg(1),
            cube_dim.num_elems() as usize,
            line_size,
            num_lines_per_unit,
            input_elem,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(ReduceError::Launch(err)),
    }
}

#[cube(launch_unchecked)]
fn shared_sum_kernel<N: Numeric>(
    input: &LinearView<Line<N>>,
    output: &mut Tensor<Atomic<N>>,
    #[comptime] shared_memory_size: usize,
    #[comptime] line_size: LineSize,
    #[comptime] num_lines_per_unit: usize,
    #[define(N)] _dtype: ElemType,
) {
    let mut shared_memory = SharedMemory::new_lined(shared_memory_size, line_size);
    shared_memory[UNIT_POS as usize] = Line::empty(line_size).fill(N::from_int(0));

    // Each unit reduce `num_lines_per_unit` lines.
    let start = ABSOLUTE_POS * num_lines_per_unit;
    let end = start + num_lines_per_unit;

    // Prevent out-of-bound access
    let start = select(start < input.shape(), start, input.shape());
    let end = select(end < input.shape(), end, input.shape());

    // Each unit sum its lines.
    for k in start..end {
        shared_memory[UNIT_POS as usize] += input[k];
    }

    // Sum all lines within the shared_memory to a single line.
    let line = sum_shared_memory(&mut shared_memory);

    // Sum all the elements within the line.
    let sum = RuntimeCell::<N>::new(N::from_int(0));
    #[unroll]
    for k in 0..line_size {
        let update = line[k] + sum.read();
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
fn sum_shared_memory<N: Numeric>(accumulator: &mut SharedMemory<Line<N>>) -> Line<N> {
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
