use cubecl::{Runtime, client::ComputeClient, prelude::TensorHandleRef};

use crate::{
    components::{
        global::read::{
            async_full_cyclic, async_full_strided, async_partial_cyclic::AsyncPartialCyclicLoading,
            async_partial_strided::AsyncPartialStridedLoading, sync_full_strided,
            sync_full_tilewise,
        },
        stage::{ColMajorTilingOrder, RowMajorTilingOrder},
        tile::{cmma::CmmaMatmul, io::Filled, mma::MmaMatmul},
    },
    definition::{MatmulElems, MatmulSetupError},
    launch::{handle::MatmulInputHandleRef, launch_naive, launch2},
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

type Cmma = CmmaMatmul<Filled>;
type Mma = MmaMatmul;

#[derive(Debug, Clone, Default)]
pub enum Strategy {
    SimpleCyclicCmma(BlueprintStrategy<SimpleAlgorithm<Cmma>>),
    SimpleCyclicMma(BlueprintStrategy<SimpleAlgorithm<Mma>>),
    SimpleStridedCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleStridedMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleTilewiseCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTilewiseMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleAsyncStridedCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
            >,
        >,
    ),
    SimpleAsyncStridedMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                async_full_strided::AsyncFullStridedLoading,
                async_full_strided::AsyncFullStridedLoading,
            >,
        >,
    ),
    SimpleAsyncCyclicCmma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Cmma,
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleAsyncCyclicMma(
        BlueprintStrategy<
            SimpleAlgorithm<
                Mma,
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTmaCmma(BlueprintStrategy<SimpleTmaAlgorithm<Cmma>>),
    SimpleTmaMma(BlueprintStrategy<SimpleTmaAlgorithm<Mma>>),
    DoubleCyclicCmma(BlueprintStrategy<CyclicDoubleBufferingAlgorithm<Cmma>>),
    DoubleCyclicMma(BlueprintStrategy<CyclicDoubleBufferingAlgorithm<Mma>>),
    DoubleTilewiseCmma(BlueprintStrategy<TilewiseDoubleBufferingAlgorithm<Cmma>>),
    DoubleTilewiseMma(BlueprintStrategy<TilewiseDoubleBufferingAlgorithm<Mma>>),
    DoubleHybridCmma(BlueprintStrategy<HybridDoubleBufferingAlgorithm<Cmma>>),
    DoubleHybridMma(BlueprintStrategy<HybridDoubleBufferingAlgorithm<Mma>>),
    DoubleAsyncCyclicCmma(BlueprintStrategy<AsyncCyclicDoubleBufferingAlgorithm<Cmma>>),
    DoubleAsyncCyclicMma(BlueprintStrategy<AsyncCyclicDoubleBufferingAlgorithm<Mma>>),
    DoubleAsyncStridedCmma(BlueprintStrategy<AsyncStridedDoubleBufferingAlgorithm<Cmma>>),
    DoubleAsyncStridedMma(BlueprintStrategy<AsyncStridedDoubleBufferingAlgorithm<Mma>>),
    DoubleTmaCmma(BlueprintStrategy<TmaDoubleBufferingAlgorithm<Cmma>>),
    DoubleTmaMma(BlueprintStrategy<TmaDoubleBufferingAlgorithm<Mma>>),
    SpecializedCyclicCmma(
        BlueprintStrategy<
            SpecializedAlgorithm<Cmma, AsyncPartialCyclicLoading<ColMajorTilingOrder>>,
        >,
    ),
    SpecializedCyclicMma(
        BlueprintStrategy<
            SpecializedAlgorithm<Mma, AsyncPartialCyclicLoading<ColMajorTilingOrder>>,
        >,
    ),
    SpecializedStridedCmma(
        BlueprintStrategy<SpecializedAlgorithm<Cmma, AsyncPartialStridedLoading>>,
    ),
    SpecializedStridedMma(BlueprintStrategy<SpecializedAlgorithm<Mma, AsyncPartialStridedLoading>>),
    SpecializedTmaCmma(BlueprintStrategy<SpecializedAlgorithm<Cmma>>),
    SpecializedTmaMma(BlueprintStrategy<SpecializedAlgorithm<Mma>>),
    OrderedDoubleCmma(BlueprintStrategy<OrderedDoubleBufferingAlgorithm<Cmma>>),
    OrderedDoubleMma(BlueprintStrategy<OrderedDoubleBufferingAlgorithm<Mma>>),
    SimpleUnit(BlueprintStrategy<SimpleUnitAlgorithm>),
    DoubleUnit(BlueprintStrategy<DoubleUnitAlgorithm>),
    SimpleVecMat(BlueprintStrategy<SimpleVecMatAlgorithm>),
    DoubleVecMat(BlueprintStrategy<DoubleVecMatAlgorithm>),
    Naive,
    #[default]
    Auto,
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
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleCyclicMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleStridedCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleStridedMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTilewiseCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTilewiseMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncStridedCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncStridedMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncCyclicCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleAsyncCyclicMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTmaCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleTmaMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleCyclicCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleCyclicMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTilewiseCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTilewiseMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleHybridCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleHybridMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncCyclicCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncCyclicMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncStridedCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleAsyncStridedMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTmaCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleTmaMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedCyclicCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedCyclicMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedStridedCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedStridedMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedTmaCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SpecializedTmaMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::OrderedDoubleCmma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::OrderedDoubleMma(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleUnit(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleUnit(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::SimpleVecMat(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
            }
            Strategy::DoubleVecMat(selection) => {
                launch2::launch_ref(client, lhs, rhs, out, selection, dtypes)
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
