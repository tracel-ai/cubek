use crate::{error::PrefixScanError, eval::strategy::PrefixScanStrategy};
use cubecl::{
    CubeDim,
    client::Client,
    prelude::*,
    tensor_vector_size_parallel,
    zspace::{Shape, Strides},
};
use cubek_std::cube_count::cube_count_spread_with_total;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum PrefixScanKind {
    /// Combines all preceeding elements, but not the element itself
    Exclusive,
    /// Combines all preceeding elements and the element itself
    Inclusive,
}

impl PrefixScanKind {
    #[cfg(feature = "benchmarks")]
    pub const ALL: &[PrefixScanKind] = &[PrefixScanKind::Exclusive, PrefixScanKind::Inclusive];

    pub fn label(self) -> &'static str {
        match self {
            PrefixScanKind::Exclusive => "Exclusive",
            PrefixScanKind::Inclusive => "Inclusive",
        }
    }
}

pub(crate) struct PrefixScanSettings {
    pub vector_size: u32,
    pub units_per_cube: u32,
}

impl PrefixScanSettings {
    pub(crate) fn new(
        properties: &DeviceProperties,
        dtype: ElemType,
        shape: &Shape,
        strides: &Strides,
    ) -> Self {
        let vector_size = tensor_vector_size_parallel(
            properties.vector_sizes_in_registers(dtype.size(), 1),
            shape,
            strides,
            0,
        ) as u32;
        let units_per_cube = if properties.hardware.num_cpu_cores.is_some() {
            // Synchronizing CPU cores inside a cube is very inefficient.
            // Better to assign them different cubes instead.
            1
        } else {
            properties.hardware.max_units_per_cube
        };
        Self {
            vector_size,
            units_per_cube,
        }
    }
}

pub(crate) struct PrefixScanLaunchSettings {
    pub working_cubes: usize,
    // pub launched_cubes: usize,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
}

impl PrefixScanLaunchSettings {
    pub(crate) fn new(
        client: &Client,
        shape: &Shape,
        settings: &PrefixScanSettings,
    ) -> Result<Self, PrefixScanError> {
        let working_cubes =
            shape[0].div_ceil((settings.units_per_cube * settings.vector_size) as usize);
        let (cube_count, launched_cubes) = cube_count_spread_with_total(client, working_cubes);
        if working_cubes != launched_cubes {
            return Err(PrefixScanError::CubeCountCanNotBeFactored);
        }
        if launched_cubes > u32::MAX as usize {
            return Err(PrefixScanError::CubeCountOverflowsU32);
        }
        Ok(Self {
            working_cubes,
            // launched_cubes,
            cube_count,
            cube_dim: CubeDim::new_3d(settings.units_per_cube, 1, 1),
        })
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PrefixScanBlueprint {
    pub units_per_cube: u32,
    pub cube_scan_size: u32,
    pub cube_scan_size_log: u32,
    pub plane_size: u32,
    pub plane_size_log: u32,
    pub plane_sized_layer_count: u32,
    pub plane_scan_before_cube_scan: bool,
    pub use_task_allocator: bool,
    pub scan_kind: PrefixScanKind,
    pub strategy: PrefixScanStrategy,
}

impl PrefixScanBlueprint {
    pub(crate) fn new(
        properties: &DeviceProperties,
        scan_kind: PrefixScanKind,
        settings: &PrefixScanSettings,
        strategy: PrefixScanStrategy,
    ) -> Result<Self, PrefixScanError> {
        // https://dl.acm.org/doi/10.1145/3485508
        // - LOBE (Linear OBE): Nvidia and AMD
        // - HSA (Heterogeneous System Architecture): Apple
        let use_task_allocator =
            properties.identity.name.contains("Nvidia") || properties.identity.name.contains("AMD");
        let plane_scan_before_cube_scan = false;
        let plane_size = properties.hardware.plane_size_max;
        let mut cube_scan_size = settings.units_per_cube;
        if plane_scan_before_cube_scan {
            cube_scan_size /= plane_size;
        }
        let plane_sized_layer_count = if plane_size == 1 {
            0
        } else {
            cube_scan_size.ilog2().div_ceil(plane_size.ilog2())
        };
        if !properties.features.device_memory_scope {
            return Err(PrefixScanError::MissingDeviceMemoryScope);
        }
        Ok(Self {
            units_per_cube: settings.units_per_cube,
            cube_scan_size,
            cube_scan_size_log: cube_scan_size.ilog2(),
            plane_size,
            plane_size_log: plane_size.ilog2(),
            plane_sized_layer_count,
            use_task_allocator,
            plane_scan_before_cube_scan,
            scan_kind,
            strategy,
        })
    }
}
