use crate::error::PrefixScanError;
use cubecl::{prelude::*, zspace::Shape};
use cubek_std::cube_count::cube_count_spread_with_total;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum PrefixScanKind {
    /// Combines all preceeding elements, but not the element itself
    Exclusive,
    /// Combines all preceeding elements and the element itself
    Inclusive,
}

impl PrefixScanKind {
    pub const ALL: &[PrefixScanKind] = &[PrefixScanKind::Exclusive, PrefixScanKind::Inclusive];

    pub fn label(self) -> &'static str {
        match self {
            PrefixScanKind::Exclusive => "Exclusive",
            PrefixScanKind::Inclusive => "Inclusive",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum PrefixScanLocalStrategy {
    /// Low Latency
    // 1973: https://dl.acm.org/doi/10.1109/TC.1973.5009159
    // 1986: https://dl.acm.org/doi/10.1145/7902.7903
    KoggeStoneOrHillisSteele,
    /// High Efficiency
    // 1980: https://dl.acm.org/doi/10.1145/322217.322232
    LadnerFischer,
    /// High Throughput
    // 1982: https://dl.acm.org/doi/10.1109/TC.1982.1675982
    BrentKung,
}

impl PrefixScanLocalStrategy {
    pub const ALL: &[PrefixScanLocalStrategy] = &[
        PrefixScanLocalStrategy::KoggeStoneOrHillisSteele,
        PrefixScanLocalStrategy::LadnerFischer,
        PrefixScanLocalStrategy::BrentKung,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PrefixScanLocalStrategy::KoggeStoneOrHillisSteele => "Kogge-Stone",
            PrefixScanLocalStrategy::LadnerFischer => "Ladner-Fischer",
            PrefixScanLocalStrategy::BrentKung => "Brent-Kung",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum PrefixScanGlobalStrategy {
    // 2013: https://dl.acm.org/doi/10.1145/2517327.2442539
    StreamScan,
    // 2016: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
    DecoupledLookback,
}

impl PrefixScanGlobalStrategy {
    pub const ALL: &[PrefixScanGlobalStrategy] = &[
        PrefixScanGlobalStrategy::StreamScan,
        PrefixScanGlobalStrategy::DecoupledLookback,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PrefixScanGlobalStrategy::StreamScan => "Stream-Scan",
            PrefixScanGlobalStrategy::DecoupledLookback => "Decoupled-Lookback",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PrefixScanStrategy {
    pub cube: PrefixScanLocalStrategy,
    pub global: PrefixScanGlobalStrategy,
}

impl core::fmt::Display for PrefixScanStrategy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}-{}", self.cube.label(), self.global.label())
    }
}

pub fn strategies() -> Vec<PrefixScanStrategy> {
    let mut strategies = Vec::new();
    for &cube in PrefixScanLocalStrategy::ALL {
        for &global in PrefixScanGlobalStrategy::ALL {
            strategies.push(PrefixScanStrategy { cube, global });
        }
    }
    strategies
}

pub struct PrefixScanSettings {
    pub vector_size: u32,
    pub units_per_cube: u32,
    pub strategy: PrefixScanStrategy,
}

impl PrefixScanSettings {
    pub fn new(
        properties: &DeviceProperties,
        dtype: ElemType,
        mut strategy: PrefixScanStrategy,
    ) -> Self {
        let vector_size = properties
            .vector_sizes_in_registers(dtype.size(), 1)
            .next()
            .unwrap() as u32;
        let units_per_cube = if properties.hardware.num_cpu_cores.is_some() {
            strategy.cube = PrefixScanLocalStrategy::KoggeStoneOrHillisSteele;
            // Synchronizing CPU cores inside a cube is very inefficient.
            // Better to assign them different cubes instead.
            1
        } else {
            properties.hardware.max_units_per_cube
        };
        Self {
            vector_size,
            units_per_cube,
            strategy,
        }
    }
}

impl core::fmt::Display for PrefixScanSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}-{}-{}",
            self.vector_size, self.units_per_cube, self.strategy
        )
    }
}

pub struct PrefixScanLaunchSettings {
    pub working_cubes: usize,
    pub launched_cubes: usize,
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
}

impl PrefixScanLaunchSettings {
    pub fn new(
        client: &Client,
        shape: &Shape,
        settings: &PrefixScanSettings,
    ) -> Result<Self, PrefixScanError> {
        let working_cubes =
            shape[0].div_ceil((settings.units_per_cube * settings.vector_size) as usize);
        let (cube_count, launched_cubes) = cube_count_spread_with_total(client, working_cubes);
        if launched_cubes > u32::MAX as usize {
            return Err(PrefixScanError::CubeCountOverflowsU32);
        }
        if shape[0].rem_euclid(settings.vector_size as usize) > 0 {
            return Err(PrefixScanError::InputSizeNotMultipleOfVectorSize);
        }
        Ok(Self {
            working_cubes,
            launched_cubes,
            cube_count,
            cube_dim: CubeDim::new_3d(settings.units_per_cube, 1, 1),
        })
    }
}
