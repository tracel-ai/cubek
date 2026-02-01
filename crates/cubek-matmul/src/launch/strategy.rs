use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use crate::{
    components::{
        global::read::{
            async_full_cyclic, async_full_strided, async_partial_cyclic::AsyncPartialCyclicLoading,
            async_partial_strided::AsyncPartialStridedLoading, sync_full_strided,
            sync_full_tilewise,
        },
        stage::{ColMajorTilingOrder, RowMajorTilingOrder},
        tile::{cmma::CmmaMatmul, mma::MmaMatmul},
    },
    definition::{MatmulElems, MatmulSetupError},
    launch::{handle::MatmulInputHandleRef, launch_naive, launch_tiling},
    routines::{
        BlueprintStrategy,
        double_buffering::{
            AsyncCyclicDoubleBufferingAlgorithm, AsyncStridedDoubleBufferingAlgorithm,
            CyclicDoubleBufferingAlgorithm, HybridDoubleBufferingAlgorithm,
            TilewiseDoubleBufferingAlgorithm, TmaDoubleBufferingAlgorithm,
        },
        double_unit::DoubleUnitAlgorithm,
        ordered_double_buffering::OrderedDoubleBufferingAlgorithm,
        simple::{SimpleAlgorithm, SimpleTmaAlgorithm},
        simple_unit::SimpleUnitAlgorithm,
        specialized::SpecializedAlgorithm,
        vecmat::{DoubleVecMatAlgorithm, SimpleVecMatAlgorithm},
    },
};

type Cmma = CmmaMatmul;
type Mma = MmaMatmul;

#[allow(clippy::type_complexity)]
#[derive(Clone, Default)]
pub enum Strategy {
    SimpleCyclicCmma(BlueprintStrategy<(), SimpleAlgorithm<Cmma>>),
    SimpleCyclicMma(BlueprintStrategy<(), SimpleAlgorithm<Mma>>),
    SimpleStridedCmma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Cmma,
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleStridedMma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Mma,
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleTilewiseCmma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Cmma,
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTilewiseMma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Mma,
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleAsyncStridedCmma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Cmma,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
            >,
        >,
    ),
    SimpleAsyncStridedMma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Mma,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
            >,
        >,
    ),
    SimpleAsyncCyclicCmma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Cmma,
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleAsyncCyclicMma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                Mma,
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTmaCmma(BlueprintStrategy<(), SimpleTmaAlgorithm<Cmma>>),
    SimpleTmaMma(BlueprintStrategy<(), SimpleTmaAlgorithm<Mma>>),
    DoubleCyclicCmma(BlueprintStrategy<(), CyclicDoubleBufferingAlgorithm<Cmma>>),
    DoubleCyclicMma(BlueprintStrategy<(), CyclicDoubleBufferingAlgorithm<Mma>>),
    DoubleTilewiseCmma(BlueprintStrategy<(), TilewiseDoubleBufferingAlgorithm<Cmma>>),
    DoubleTilewiseMma(BlueprintStrategy<(), TilewiseDoubleBufferingAlgorithm<Mma>>),
    DoubleHybridCmma(BlueprintStrategy<(), HybridDoubleBufferingAlgorithm<Cmma>>),
    DoubleHybridMma(BlueprintStrategy<(), HybridDoubleBufferingAlgorithm<Mma>>),
    DoubleAsyncCyclicCmma(BlueprintStrategy<(), AsyncCyclicDoubleBufferingAlgorithm<Cmma>>),
    DoubleAsyncCyclicMma(BlueprintStrategy<(), AsyncCyclicDoubleBufferingAlgorithm<Mma>>),
    DoubleAsyncStridedCmma(BlueprintStrategy<(), AsyncStridedDoubleBufferingAlgorithm<Cmma>>),
    DoubleAsyncStridedMma(BlueprintStrategy<(), AsyncStridedDoubleBufferingAlgorithm<Mma>>),
    DoubleTmaCmma(BlueprintStrategy<(), TmaDoubleBufferingAlgorithm<Cmma>>),
    DoubleTmaMma(BlueprintStrategy<(), TmaDoubleBufferingAlgorithm<Mma>>),
    SpecializedCyclicCmma(
        BlueprintStrategy<
            (),
            SpecializedAlgorithm<Cmma, AsyncPartialCyclicLoading<ColMajorTilingOrder>>,
        >,
    ),
    SpecializedCyclicMma(
        BlueprintStrategy<
            (),
            SpecializedAlgorithm<Mma, AsyncPartialCyclicLoading<ColMajorTilingOrder>>,
        >,
    ),
    SpecializedStridedCmma(
        BlueprintStrategy<(), SpecializedAlgorithm<Cmma, AsyncPartialStridedLoading>>,
    ),
    SpecializedStridedMma(
        BlueprintStrategy<(), SpecializedAlgorithm<Mma, AsyncPartialStridedLoading>>,
    ),
    SpecializedTmaCmma(BlueprintStrategy<(), SpecializedAlgorithm<Cmma>>),
    SpecializedTmaMma(BlueprintStrategy<(), SpecializedAlgorithm<Mma>>),
    OrderedDoubleCmma(BlueprintStrategy<(), OrderedDoubleBufferingAlgorithm<Cmma>>),
    OrderedDoubleMma(BlueprintStrategy<(), OrderedDoubleBufferingAlgorithm<Mma>>),
    SimpleUnit(BlueprintStrategy<(), SimpleUnitAlgorithm>),
    DoubleUnit(BlueprintStrategy<(), DoubleUnitAlgorithm>),
    SimpleVecMat(BlueprintStrategy<(), SimpleVecMatAlgorithm>),
    DoubleVecMat(BlueprintStrategy<(), DoubleVecMatAlgorithm>),
    Naive,
    #[default]
    Auto,
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::SimpleCyclicCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_cyclic_cmma{}",
                blueprint_strategy
            )),
            Strategy::SimpleCyclicMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_cyclic_mma{}",
                blueprint_strategy
            )),
            Strategy::SimpleStridedCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_strided_cmma{}",
                blueprint_strategy
            )),
            Strategy::SimpleStridedMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_strided_mma{}",
                blueprint_strategy
            )),
            Strategy::SimpleTilewiseCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_tilewise_cmma{}",
                blueprint_strategy
            )),
            Strategy::SimpleTilewiseMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_tilewise_mma{}",
                blueprint_strategy
            )),
            Strategy::SimpleAsyncStridedCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_async_strided_cmma{}",
                blueprint_strategy
            )),
            Strategy::SimpleAsyncStridedMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_async_strided_mma{}",
                blueprint_strategy
            )),
            Strategy::SimpleAsyncCyclicCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_async_cyclic_cmma{}",
                blueprint_strategy
            )),
            Strategy::SimpleAsyncCyclicMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_simple_async_cyclic_mma{}",
                blueprint_strategy
            )),
            Strategy::SimpleTmaCmma(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_simple_tma_cmma{}", blueprint_strategy))
            }
            Strategy::SimpleTmaMma(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_simple_tma_mma{}", blueprint_strategy))
            }
            Strategy::DoubleCyclicCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_cyclic_cmma{}",
                blueprint_strategy
            )),
            Strategy::DoubleCyclicMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_cyclic_mma{}",
                blueprint_strategy
            )),
            Strategy::DoubleTilewiseCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_tilewise_cmma{}",
                blueprint_strategy
            )),
            Strategy::DoubleTilewiseMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_tilewise_mma{}",
                blueprint_strategy
            )),
            Strategy::DoubleHybridCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_hybrid_cmma{}",
                blueprint_strategy
            )),
            Strategy::DoubleHybridMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_hybrid_mma{}",
                blueprint_strategy
            )),
            Strategy::DoubleAsyncCyclicCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_async_cyclic_cmma{}",
                blueprint_strategy
            )),
            Strategy::DoubleAsyncCyclicMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_async_cyclic_mma{}",
                blueprint_strategy
            )),
            Strategy::DoubleAsyncStridedCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_async_strided_cmma{}",
                blueprint_strategy
            )),
            Strategy::DoubleAsyncStridedMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_double_async_strided_mma{}",
                blueprint_strategy
            )),
            Strategy::DoubleTmaCmma(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_double_tma_cmma{}", blueprint_strategy))
            }
            Strategy::DoubleTmaMma(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_double_tma_mma{}", blueprint_strategy))
            }
            Strategy::SpecializedCyclicCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_specialized_cyclic_cmma{}",
                blueprint_strategy
            )),
            Strategy::SpecializedCyclicMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_specialized_cyclic_mma{}",
                blueprint_strategy
            )),
            Strategy::SpecializedStridedCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_specialized_strided_cmma{}",
                blueprint_strategy
            )),
            Strategy::SpecializedStridedMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_specialized_strided_mma{}",
                blueprint_strategy
            )),
            Strategy::SpecializedTmaCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_specialized_tma_cmma{}",
                blueprint_strategy
            )),
            Strategy::SpecializedTmaMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_specialized_tma_mma{}",
                blueprint_strategy
            )),
            Strategy::OrderedDoubleCmma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_ordered_double_cmma{}",
                blueprint_strategy
            )),
            Strategy::OrderedDoubleMma(blueprint_strategy) => f.write_fmt(format_args!(
                "matmul_ordered_double_mma{}",
                blueprint_strategy
            )),
            Strategy::SimpleUnit(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_simple_unit{}", blueprint_strategy))
            }
            Strategy::DoubleUnit(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_double_unit{}", blueprint_strategy))
            }
            Strategy::SimpleVecMat(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_simple_vecmat{}", blueprint_strategy))
            }
            Strategy::DoubleVecMat(blueprint_strategy) => {
                f.write_fmt(format_args!("matmul_double_vecmat{}", blueprint_strategy))
            }
            Strategy::Naive => f.write_str("matmul_naive"),
            Strategy::Auto => f.write_str("matmul_auto"),
        }
    }
}

#[allow(clippy::result_large_err)]
impl Strategy {
    pub(crate) fn launch_ref<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs: &MatmulInputHandleRef<R>,
        rhs: &MatmulInputHandleRef<R>,
        out: &TensorHandleRef<R>,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        match self {
            Strategy::SimpleCyclicCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleCyclicMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleStridedCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleStridedMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTilewiseCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTilewiseMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncStridedCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncStridedMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncCyclicCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncCyclicMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTmaCmma(selection) => {
                launch_tiling::launch_ref_tma(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTmaMma(selection) => {
                launch_tiling::launch_ref_tma(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleCyclicCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleCyclicMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTilewiseCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTilewiseMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleHybridCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleHybridMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncCyclicCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncCyclicMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncStridedCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncStridedMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTmaCmma(selection) => {
                launch_tiling::launch_ref_tma(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTmaMma(selection) => {
                launch_tiling::launch_ref_tma(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedCyclicCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedCyclicMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedStridedCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedStridedMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedTmaCmma(selection) => {
                launch_tiling::launch_ref_tma(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedTmaMma(selection) => {
                launch_tiling::launch_ref_tma(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::OrderedDoubleCmma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::OrderedDoubleMma(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleUnit(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleUnit(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleVecMat(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleVecMat(selection) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::Naive => launch_naive::launch_ref(client, lhs, rhs, out, dtypes),
            Strategy::Auto => auto(client, lhs, rhs, out, dtypes),
        }
    }
}

fn auto<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    if let Err(err) =
        Strategy::SimpleCyclicCmma(Default::default()).launch_ref(client, lhs, rhs, out, dtypes)
    {
        match err {
            MatmulSetupError::Unavailable(_) => {
                Strategy::SimpleUnit(Default::default())
                    .launch_ref(client, lhs, rhs, out, dtypes)
                    .unwrap();
            }
            _ => panic!("{err:?}"),
        }
    }

    Ok(())
}
