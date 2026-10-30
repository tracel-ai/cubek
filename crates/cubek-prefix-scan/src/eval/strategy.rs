#[cfg(feature = "benchmarks")]
use cubek_test_utils::CatalogEntry;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum CubeScanStrategy {
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

impl CubeScanStrategy {
    #[cfg(feature = "benchmarks")]
    const ALL: &[CubeScanStrategy] = &[
        CubeScanStrategy::KoggeStoneOrHillisSteele,
        CubeScanStrategy::LadnerFischer,
        CubeScanStrategy::BrentKung,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            CubeScanStrategy::KoggeStoneOrHillisSteele => "Kogge-Stone",
            CubeScanStrategy::LadnerFischer => "Ladner-Fischer",
            CubeScanStrategy::BrentKung => "Brent-Kung",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum GlobalScanStrategy {
    // 2013: https://dl.acm.org/doi/10.1145/2517327.2442539
    StreamScan,
    // 2016: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
    DecoupledLookback,
}

impl GlobalScanStrategy {
    #[cfg(feature = "benchmarks")]
    const ALL: &[GlobalScanStrategy] = &[
        GlobalScanStrategy::StreamScan,
        GlobalScanStrategy::DecoupledLookback,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            GlobalScanStrategy::StreamScan => "Stream-Scan",
            GlobalScanStrategy::DecoupledLookback => "Decoupled-Lookback",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct ScanStrategy {
    pub cube: CubeScanStrategy,
    pub global: GlobalScanStrategy,
}

#[cfg(feature = "benchmarks")]
pub fn strategies() -> Vec<CatalogEntry<ScanStrategy>> {
    let mut strategies = Vec::new();
    for &cube in CubeScanStrategy::ALL {
        for &global in GlobalScanStrategy::ALL {
            let name = format!("{} & {}", cube.label(), global.label());
            strategies.push(CatalogEntry::new(
                name.to_lowercase(),
                name,
                ScanStrategy { cube, global },
            ));
        }
    }
    strategies
}
