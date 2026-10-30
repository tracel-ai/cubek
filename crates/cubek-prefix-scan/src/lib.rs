use crate::{
    eval::strategy::{GlobalScanStrategy, ScanStrategy},
    instructions::{Maximum, Minimum, Product, ScanInstructionKind, Sum},
};
use cubecl::{
    CubeDim, client::Client, prelude::*, std::tensor::TensorHandle, tensor_vector_size_parallel,
};
use cubek_std::cube_count::cube_count_spread_with_total;
use error::PrefixScanError;
use instructions::ScanInstruction;

pub mod error;
pub mod eval;
pub mod instructions;
pub mod kernels;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum ScanKind {
    /// Combines all preceeding elements, but not the element itself
    Exclusive,
    /// Combines all preceeding elements and the element itself
    Inclusive,
}

impl ScanKind {
    #[cfg(feature = "benchmarks")]
    pub const ALL: &[ScanKind] = &[ScanKind::Exclusive, ScanKind::Inclusive];

    pub fn label(self) -> &'static str {
        match self {
            ScanKind::Exclusive => "Exclusive",
            ScanKind::Inclusive => "Inclusive",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct Blueprint {
    pub units_per_cube: u32,
    pub cube_scan_size: u32,
    pub cube_scan_size_log: u32,
    pub plane_size: u32,
    pub plane_size_log: u32,
    pub plane_sized_layer_count: u32,
    pub plane_scan_before_cube_scan: bool,
    pub use_task_allocator: bool,
    pub scan_kind: ScanKind,
    pub strategy: ScanStrategy,
}

pub fn prefix_scan_launch_generic_instruction<I: ScanInstruction>(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
    scan_kind: ScanKind,
    strategy: ScanStrategy,
) -> Result<(), PrefixScanError> {
    if input.shape.rank() != 1 {
        return Err(PrefixScanError::UnsupportedRank {
            rank: input.shape.rank(),
        });
    }
    if input.shape != output.shape {
        return Err(PrefixScanError::InputOutputShapeMismatch {
            input: input.shape.to_vec(),
            output: output.shape.to_vec(),
        });
    }
    if !client.features().device_memory_scope {
        return Err(PrefixScanError::MissingDeviceMemoryScope);
    }

    // https://dl.acm.org/doi/10.1145/3485508
    // - LOBE (Linear OBE): Nvidia and AMD
    // - HSA (Heterogeneous System Architecture): Apple
    let use_task_allocator = client.properties().identity.name.contains("Nvidia")
        || client.properties().identity.name.contains("AMD");

    let plane_scan_before_cube_scan = false;
    let vector_size = tensor_vector_size_parallel(
        client
            .properties()
            .vector_sizes_in_registers(dtype.size(), 1),
        &input.shape,
        &input.strides,
        0,
    ) as u32;
    let plane_size = client.properties().hardware.plane_size_max;
    let units_per_cube = if client.properties().hardware.num_cpu_cores.is_some() {
        // Synchronizing CPU cores inside a cube is very inefficient.
        // Better to assign them different cubes instead.
        1
    } else {
        client.properties().hardware.max_units_per_cube
    };
    let mut cube_scan_size = units_per_cube;
    if plane_scan_before_cube_scan {
        cube_scan_size /= plane_size;
    }
    let plane_sized_layer_count = if plane_size == 1 {
        0
    } else {
        cube_scan_size.ilog2().div_ceil(plane_size.ilog2())
    };
    let working_cubes = input.shape[0].div_ceil((units_per_cube * vector_size) as usize);
    let (cube_count, launched_cubes) = cube_count_spread_with_total(client, working_cubes);
    if working_cubes != launched_cubes {
        return Err(PrefixScanError::CubeCountCanNotBeFactored);
    }
    if launched_cubes > u32::MAX as usize {
        return Err(PrefixScanError::CubeCountOverflowsU32);
    }

    let reductions = match strategy.global {
        GlobalScanStrategy::StreamScan => None,
        GlobalScanStrategy::DecoupledLookback => {
            Some(TensorHandle::empty(client, [working_cubes], dtype))
        }
    };
    let task_allocator = TensorHandle::zeros(client, [1], u32::elem_type_native());
    let task_status = TensorHandle::zeros(client, [working_cubes], u32::elem_type_native());

    unsafe {
        kernels::single_pass_prefix_scan::launch_unchecked::<I>(
            client,
            cube_count,
            CubeDim::new_3d(units_per_cube, 1, 1),
            vector_size as usize,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            output.clone().into_tensor_arg(),
            reductions
                .map(|tensor_handle| tensor_handle.into_arg())
                .unwrap_or_else(|| output.clone().into_tensor_arg()),
            task_allocator.into_arg(),
            task_status.into_arg(),
            Blueprint {
                units_per_cube,
                cube_scan_size,
                cube_scan_size_log: cube_scan_size.ilog2(),
                plane_size,
                plane_size_log: plane_size.ilog2(),
                plane_sized_layer_count,
                use_task_allocator,
                plane_scan_before_cube_scan,
                scan_kind,
                strategy,
            },
            dtype,
        );
    }
    Ok(())
}

pub fn prefix_scan_launch(
    client: &Client,
    input: TensorBinding,
    output: TensorBinding,
    dtype: ElemType,
    scan_kind: ScanKind,
    scan_instruction_kind: ScanInstructionKind,
    strategy: ScanStrategy,
) -> Result<(), PrefixScanError> {
    match scan_instruction_kind {
        ScanInstructionKind::Sum => prefix_scan_launch_generic_instruction::<Sum>(
            client, input, output, dtype, scan_kind, strategy,
        ),
        ScanInstructionKind::Product => prefix_scan_launch_generic_instruction::<Product>(
            client, input, output, dtype, scan_kind, strategy,
        ),
        ScanInstructionKind::Minimum => prefix_scan_launch_generic_instruction::<Minimum>(
            client, input, output, dtype, scan_kind, strategy,
        ),
        ScanInstructionKind::Maximum => prefix_scan_launch_generic_instruction::<Maximum>(
            client, input, output, dtype, scan_kind, strategy,
        ),
    }
}
