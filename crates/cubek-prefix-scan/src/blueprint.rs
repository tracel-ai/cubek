use crate::{
    base::{PrefixScanKind, PrefixScanSettings, PrefixScanStrategy},
    error::PrefixScanError,
};
use cubecl::{ir::PciVendor, prelude::*};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PrefixScanBlueprint {
    pub cube_scan_size_ilog2: u32,
    pub use_task_allocator: bool,
    pub scan_kind: PrefixScanKind,
    pub strategy: PrefixScanStrategy,
}

impl PrefixScanBlueprint {
    pub(crate) fn new(
        properties: &DeviceProperties,
        scan_kind: PrefixScanKind,
        settings: &PrefixScanSettings,
    ) -> Result<Self, PrefixScanError> {
        // https://dl.acm.org/doi/10.1145/3485508
        // - LOBE (Linear OBE): NVIDIA and AMD
        // - HSA (Heterogeneous System Architecture): Apple
        let use_task_allocator = if let Some(vendor) = properties
            .identity
            .physical
            .as_ref()
            .and_then(|physical| physical.vendor)
        {
            matches!(vendor, PciVendor::Nvidia | PciVendor::Amd)
        } else {
            properties.identity.name.contains("NVIDIA") || properties.identity.name.contains("AMD")
        };
        if 1 << settings.vector_size.ilog2() != settings.vector_size {
            return Err(PrefixScanError::VectorSizeNotPoT);
        }
        if 1 << settings.units_per_cube.ilog2() != settings.units_per_cube {
            return Err(PrefixScanError::UnitsPerCubeNotPoT);
        }
        if !properties.features.device_memory_scope {
            return Err(PrefixScanError::MissingDeviceMemoryScope);
        }
        Ok(Self {
            cube_scan_size_ilog2: settings.units_per_cube.ilog2(),
            use_task_allocator,
            scan_kind,
            strategy: settings.strategy.clone(),
        })
    }
}
