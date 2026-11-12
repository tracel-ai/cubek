//! A multi-level convolution routine at a blueprint the test forces, reached through
//! [`launch_ref`]: the tiling of the matmul it plans, on the tile matmul it names.

use cubecl::client::Client;
use cubek_convolution::components::ConvSetupError;
use cubek_convolution::definition::{
    BackwardDataBlueprint, BackwardWeightBlueprint, ConvBlueprint, ForwardBlueprint,
};
use cubek_convolution::{ConvAlgorithm, Strategy, launch_ref};
use cubek_matmul::definition::AccumulatorOperand;
use cubek_matmul::multi_level::stage::SwizzleModes;
use cubek_matmul::multi_level::{
    PartitionSize, StageSize, TileSize,
    components::{stage::PartitionBuffering, tile::TileMatmulKind},
    definition::{BatchMatmulBlueprint, TilingScheme},
};

use crate::convolution::{Conv, ConvRoutine, ConvTensors, Pass};

/// `algorithm` at the tiling `tiling`, on `tile_kind`'s instructions.
#[derive(Debug, Clone)]
pub(crate) struct Forced {
    pub algorithm: ConvAlgorithm,
    pub tile_kind: TileMatmulKind,
    pub tiling: TilingScheme,
}

impl Forced {
    /// The tile a matrix instruction offers on every device: `8 × 8 × 8` on Metal, `16 × 16 × 16`
    /// elsewhere.
    pub(crate) fn instruction_tile() -> TileSize {
        match cfg!(target_os = "macos") {
            true => TileSize { m: 8, n: 8, k: 8 },
            false => TileSize {
                m: 16,
                n: 16,
                k: 16,
            },
        }
    }

    /// `algorithm` at the smallest tiling: one tile a partition, two planes a stage along `m` and
    /// `n`.
    pub(crate) fn smallest(algorithm: ConvAlgorithm) -> Self {
        Self::at(algorithm, PartitionSize { m: 1, n: 1, k: 1 })
    }

    /// `algorithm` with `partition` tiles a partition, two planes a stage along `m` and `n`.
    pub(crate) fn at(algorithm: ConvAlgorithm, partition: PartitionSize) -> Self {
        Self {
            algorithm,
            tile_kind: TileMatmulKind::Cmma,
            tiling: TilingScheme::builder()
                .with_tile_size(Self::instruction_tile())
                .with_partition_size(partition)
                .with_stage_size(StageSize { m: 2, n: 2, k: 1 })
                .build()
                .unwrap(),
        }
    }
}

impl<const N: usize> ConvRoutine<N> for Forced {
    fn launch(
        &self,
        client: &Client,
        conv: &Conv<N>,
        pass: Pass,
        tensors: &ConvTensors,
    ) -> Result<(), ConvSetupError> {
        let problem = conv.problem(pass, tensors);
        let matmul = BatchMatmulBlueprint::builder(
            self.tile_kind,
            self.tiling,
            client.properties().hardware.plane_size_max,
            &problem.as_matmul_problem(AccumulatorOperand::Absent),
        )
        .shared_swizzle(SwizzleModes::default())
        .partition_buffering(PartitionBuffering::Single)
        .build();
        let dimensionality = Conv::<N>::dimensionality();
        let blueprint = match pass {
            Pass::Forward => ConvBlueprint::Forward(ForwardBlueprint {
                matmul,
                dimensionality,
                has_bias: tensors.bias.is_some(),
            }),
            Pass::BackwardData => ConvBlueprint::BackwardData(BackwardDataBlueprint {
                matmul,
                dimensionality,
            }),
            Pass::BackwardWeight => ConvBlueprint::BackwardWeight(BackwardWeightBlueprint {
                matmul,
                dimensionality,
            }),
        };
        let strategy = Strategy::Forced {
            algorithm: self.algorithm,
            blueprint,
        };
        launch_ref(
            &strategy,
            client,
            tensors.inputs(conv, pass),
            conv.args(),
            conv.matmul_elems(),
        )
    }
}
