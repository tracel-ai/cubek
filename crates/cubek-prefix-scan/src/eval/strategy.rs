#[cfg(feature = "benchmarks")]
use cubek_test_utils::CatalogEntry;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum LocalPrefixScanStrategy {
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

impl LocalPrefixScanStrategy {
    pub const ALL: &[LocalPrefixScanStrategy] = &[
        LocalPrefixScanStrategy::KoggeStoneOrHillisSteele,
        LocalPrefixScanStrategy::LadnerFischer,
        LocalPrefixScanStrategy::BrentKung,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            LocalPrefixScanStrategy::KoggeStoneOrHillisSteele => "Kogge-Stone",
            LocalPrefixScanStrategy::LadnerFischer => "Ladner-Fischer",
            LocalPrefixScanStrategy::BrentKung => "Brent-Kung",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum GlobalPrefixScanStrategy {
    // 2013: https://dl.acm.org/doi/10.1145/2517327.2442539
    StreamScan,
    // 2016: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
    DecoupledLookback,
}

impl GlobalPrefixScanStrategy {
    pub const ALL: &[GlobalPrefixScanStrategy] = &[
        GlobalPrefixScanStrategy::StreamScan,
        GlobalPrefixScanStrategy::DecoupledLookback,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            GlobalPrefixScanStrategy::StreamScan => "Stream-Scan",
            GlobalPrefixScanStrategy::DecoupledLookback => "Decoupled-Lookback",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct PrefixScanStrategy {
    pub cube: LocalPrefixScanStrategy,
    pub global: GlobalPrefixScanStrategy,
}

#[cfg(feature = "benchmarks")]
pub fn strategies() -> Vec<CatalogEntry<PrefixScanStrategy>> {
    let mut strategies = Vec::new();
    for &cube in LocalPrefixScanStrategy::ALL {
        for &global in GlobalPrefixScanStrategy::ALL {
            let name = format!("{} & {}", cube.label(), global.label());
            strategies.push(CatalogEntry::new(
                name.to_lowercase(),
                name,
                PrefixScanStrategy { cube, global },
            ));
        }
    }
    strategies
}
