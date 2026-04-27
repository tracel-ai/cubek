use cubek_matmul::definition::TilingBlueprint;

use crate::components::Dimensionality;

/// Per-operation comptime blueprint for the convolution kernel family.
///
/// The blueprint captures the minimal comptime information needed to specialize
/// the kernel: which operation (forward / data-grad / weight-grad), the matmul
/// `TilingBlueprint`, and per-operation comptime tunables. A different blueprint
/// retriggers JIT compilation, so it is kept minimal.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConvBlueprint {
    Forward(ForwardBlueprint),
    BackwardData(BackwardDataBlueprint),
    BackwardWeight(BackwardWeightBlueprint),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ForwardBlueprint {
    pub matmul: TilingBlueprint,
    pub dimensionality: Dimensionality,
    pub has_bias: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BackwardDataBlueprint {
    pub matmul: TilingBlueprint,
    pub dimensionality: Dimensionality,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BackwardWeightBlueprint {
    pub matmul: TilingBlueprint,
    pub dimensionality: Dimensionality,
}

impl ConvBlueprint {
    pub fn matmul(&self) -> &TilingBlueprint {
        match self {
            ConvBlueprint::Forward(b) => &b.matmul,
            ConvBlueprint::BackwardData(b) => &b.matmul,
            ConvBlueprint::BackwardWeight(b) => &b.matmul,
        }
    }

    pub fn dimensionality(&self) -> Dimensionality {
        match self {
            ConvBlueprint::Forward(b) => b.dimensionality,
            ConvBlueprint::BackwardData(b) => b.dimensionality,
            ConvBlueprint::BackwardWeight(b) => b.dimensionality,
        }
    }
}
