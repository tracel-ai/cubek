use std::fmt::Display;

use serde::{Deserialize, Serialize};

#[derive(Clone)]
pub struct ConvolutionArgs<const N_SPATIAL: usize> {
    pub stride: [usize; N_SPATIAL],
    pub padding: [usize; N_SPATIAL],
    pub dilation: [usize; N_SPATIAL],
}

pub enum Strategy {
    Simple {
        read_strategy: ReadingStrategy,
        tile_kind: AcceleratedTileKind,
    },
}

#[derive(Debug, Clone, Copy)]
/// Which reader to use in simple algorithms
pub enum ReadingStrategy {
    Cyclic,
    Strided,
    Tilewise,
    AsyncCyclic,
    AsyncStrided,
    Tma,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Which tile matmul to use for accelerated algorithms
pub enum AcceleratedTileKind {
    #[default]
    Cmma,
    Mma,
}

// Display implementations are used to combine and save names when autotuning.

impl Display for AcceleratedTileKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceleratedTileKind::Cmma => f.write_str("cmma"),
            AcceleratedTileKind::Mma => f.write_str("mma"),
        }
    }
}

impl Display for ReadingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReadingStrategy::Cyclic => f.write_str("cyclic"),
            ReadingStrategy::Strided => f.write_str("strided"),
            ReadingStrategy::Tilewise => f.write_str("tilewise"),
            ReadingStrategy::AsyncCyclic => f.write_str("async_cyclic"),
            ReadingStrategy::AsyncStrided => f.write_str("async_strided"),
            ReadingStrategy::Tma => f.write_str("tma"),
        }
    }
}
