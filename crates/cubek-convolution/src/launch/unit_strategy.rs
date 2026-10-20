use std::fmt::Display;

use crate::routines::unit::UnitMatmul;
use cubecl::client::Client;
use cubek_matmul::{
    definition::{MatmulElems, MatmulProblem},
    multi_level::{
        components::{stage::PartitionBuffering, tile::TileMatmulKind},
        definition::BatchMatmulBlueprint,
        routines::{
            TileSizeSelection, UnitSmemBudget, batch::simple_unit::SimpleUnitSelectionArgs,
        },
        stage::{SwizzleMode, SwizzleModes},
    },
    routine::BlueprintStrategy,
};

/// Register-tiled F32 convolution candidates. Selection is left to the caller's tuner.
///
/// A forced candidate is a fixed [`UnitTiling`]. The presets on that type span
/// the output-rows (`M`), output-width (`N`) and reduction (`K`) axes so that
/// autotune has a candidate near every regime a convolution's matmul lands in,
/// rather than only the two dense shapes the presets were first measured on.
#[derive(Clone, Copy, Debug)]
pub enum UnitStrategy {
    /// Use the general matmul selector.
    Inferred(TileSizeSelection),
    /// A fixed register tiling, one of the [`UnitTiling`] presets.
    Fixed(UnitTiling),
}

/// A forced register tiling, expressed as the tiles each unit's partition holds
/// along `(M, N, K)`.
///
/// The tile `(4, 4, 4)` and stage `(16, 8, 1)` are held constant, so the stage
/// a preset asks for is `(64*m) x (32*n) x (4*k)` output/reduction elements.
/// A preset that overflows shared memory is trimmed by [`UnitSmemBudget`], the
/// same budget the matmul unit selector fits its tilings to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitTiling {
    partition: (u32, u32, u32),
}

impl UnitTiling {
    /// `128 x 32 x 32`: few output channels, as in a 32-wide dense layer.
    pub const NARROW: Self = Self::new(2, 1, 8);
    /// `128 x 64 x 32`: a moderate output width.
    pub const BALANCED: Self = Self::new(2, 2, 8);
    /// `128 x 128 x 32`: wide outputs and the wide weight-gradient matrices a
    /// large channel count produces (`taps * C_in`), e.g. 128–512 channels.
    pub const WIDE: Self = Self::new(2, 4, 8);
    /// `128 x 64 x 64`: a long reduction, as a large `C_in * taps` contraction
    /// gives on a short feature map (the short-spatial/long-channel regime).
    pub const DEEP: Self = Self::new(2, 2, 16);
    /// `256 x 64 x 32`: many output rows, as a large spatial map times batch
    /// gives in forward and data-gradient passes.
    pub const TALL: Self = Self::new(4, 2, 8);

    const fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            partition: (m, n, k),
        }
    }
}

impl Display for UnitStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inferred(tile_size) => write!(f, "unit_{tile_size}"),
            Self::Fixed(UnitTiling {
                partition: (m, n, k),
            }) => write!(f, "unit_{m}x{n}x{k}"),
        }
    }
}

impl UnitStrategy {
    pub(crate) fn blueprint(
        self,
        client: &Client,
        problem: &MatmulProblem,
        dtypes: &MatmulElems,
    ) -> BlueprintStrategy<UnitMatmul> {
        let partition = match self {
            Self::Inferred(tile_size) => {
                return BlueprintStrategy::Inferred(SimpleUnitSelectionArgs { tile_size });
            }
            Self::Fixed(tiling) => tiling.partition,
        };
        let plane_dim = client.properties().hardware.plane_size_min.max(32);
        let tiling = UnitSmemBudget {
            dtypes,
            stage_buffering: 1,
            accumulator: problem.accumulator,
            max_smem: client.properties().hardware.max_shared_memory_size,
            plane_dim,
        }
        .fit((4, 4, 4), partition, (16, 8));
        let mut builder =
            BatchMatmulBlueprint::builder(TileMatmulKind::Register, tiling, plane_dim, problem)
                .partition_buffering(PartitionBuffering::Single);
        if client.properties().features.alignment && tiling.elements_per_stage_along_k() >= 32 {
            builder = builder.shared_swizzle(SwizzleModes {
                lhs: SwizzleMode::B128,
                rhs: SwizzleMode::B128,
                ..Default::default()
            });
        }
        BlueprintStrategy::Forced(builder.build())
    }
}
