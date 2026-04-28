use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::{
    components::{
        global::read::{
            async_full_cyclic, async_full_strided, async_partial_cyclic::AsyncPartialCyclicLoading,
            async_partial_strided::AsyncPartialStridedLoading, sync_full_strided,
            sync_full_tilewise,
        },
        stage::{ColMajorTilingOrder, RowMajorTilingOrder},
        tile_matmul::TileMatmul,
    },
    definition::{MatmulElems, MatmulSetupError},
    launch::{
        launch_naive, launch_tiling, launch_vecmat_plane_parallel, launch_vecmat_unit_perpendicular,
    },
    routines::{
        BlueprintStrategy, Routine,
        double_buffering::{
            AsyncCyclicDoubleBufferingAlgorithm, AsyncStridedDoubleBufferingAlgorithm,
            CyclicDoubleBufferingAlgorithm, DoubleBufferingArgs, HybridDoubleBufferingAlgorithm,
            TilewiseDoubleBufferingAlgorithm, TmaDoubleBufferingAlgorithm,
        },
        double_unit::DoubleUnitAlgorithm,
        ordered_double_buffering::{OrderedDoubleBufferingAlgorithm, OrderedSelectionArgs},
        simple::{SimpleAlgorithm, SimpleArgs, SimpleTmaAlgorithm},
        simple_unit::SimpleUnitAlgorithm,
        specialized::{SpecializedAlgorithm, SpecializedStrategy},
        vecmat_innerproduct::{DoubleVecMatInnerProductAlgorithm, VecMatInnerProductAlgorithm},
        vecmat_plane_parallel::GemvPlaneParallelRoutine,
        vecmat_unit_perpendicular::GemvUnitPerpendicularRoutine,
    },
};

/// Returns a clone of `sel` with `args.tile_matmul` overridden to `kind` when
/// in Inferred mode. Forced mode is left untouched since the user-supplied
/// blueprint already carries a `tile_matmul`.
fn stamp_kind<RC, A>(
    sel: &BlueprintStrategy<RC, A>,
    kind: TileMatmul,
    set: impl FnOnce(&mut A::Strategy, TileMatmul),
) -> BlueprintStrategy<RC, A>
where
    RC: crate::launch::RuntimeConfig,
    A: Routine<RC>,
{
    let mut sel = sel.clone();
    if let BlueprintStrategy::Inferred(args) = &mut sel {
        set(args, kind);
    }
    sel
}

fn set_simple(args: &mut SimpleArgs, kind: TileMatmul) {
    args.tile_matmul = kind;
}
fn set_double(args: &mut DoubleBufferingArgs, kind: TileMatmul) {
    args.tile_matmul = kind;
}
fn set_ordered(args: &mut OrderedSelectionArgs, kind: TileMatmul) {
    args.tile_matmul = kind;
}
fn set_specialized(args: &mut SpecializedStrategy, kind: TileMatmul) {
    args.tile_matmul = kind;
}

#[allow(clippy::type_complexity)]
#[derive(Clone, Default)]
pub enum Strategy {
    SimpleCyclicCmma(BlueprintStrategy<(), SimpleAlgorithm>),
    SimpleCyclicMma(BlueprintStrategy<(), SimpleAlgorithm>),
    SimpleStridedCmma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleStridedMma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                sync_full_strided::SyncFullStridedLoading,
                sync_full_strided::SyncFullStridedLoading,
            >,
        >,
    ),
    SimpleTilewiseCmma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTilewiseMma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
                sync_full_tilewise::SyncFullTilewiseLoading<ColMajorTilingOrder>,
                sync_full_tilewise::SyncFullTilewiseLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleAsyncStridedCmma(
        BlueprintStrategy<
            (),
            SimpleAlgorithm<
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
                async_full_cyclic::AsyncFullCyclicLoading<ColMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
                async_full_cyclic::AsyncFullCyclicLoading<RowMajorTilingOrder>,
            >,
        >,
    ),
    SimpleTmaCmma(BlueprintStrategy<(), SimpleTmaAlgorithm>),
    SimpleTmaMma(BlueprintStrategy<(), SimpleTmaAlgorithm>),
    DoubleCyclicCmma(BlueprintStrategy<(), CyclicDoubleBufferingAlgorithm>),
    DoubleCyclicMma(BlueprintStrategy<(), CyclicDoubleBufferingAlgorithm>),
    DoubleTilewiseCmma(BlueprintStrategy<(), TilewiseDoubleBufferingAlgorithm>),
    DoubleTilewiseMma(BlueprintStrategy<(), TilewiseDoubleBufferingAlgorithm>),
    DoubleHybridCmma(BlueprintStrategy<(), HybridDoubleBufferingAlgorithm>),
    DoubleHybridMma(BlueprintStrategy<(), HybridDoubleBufferingAlgorithm>),
    DoubleAsyncCyclicCmma(BlueprintStrategy<(), AsyncCyclicDoubleBufferingAlgorithm>),
    DoubleAsyncCyclicMma(BlueprintStrategy<(), AsyncCyclicDoubleBufferingAlgorithm>),
    DoubleAsyncStridedCmma(BlueprintStrategy<(), AsyncStridedDoubleBufferingAlgorithm>),
    DoubleAsyncStridedMma(BlueprintStrategy<(), AsyncStridedDoubleBufferingAlgorithm>),
    DoubleTmaCmma(BlueprintStrategy<(), TmaDoubleBufferingAlgorithm>),
    DoubleTmaMma(BlueprintStrategy<(), TmaDoubleBufferingAlgorithm>),
    SpecializedCyclicCmma(
        BlueprintStrategy<(), SpecializedAlgorithm<AsyncPartialCyclicLoading<ColMajorTilingOrder>>>,
    ),
    SpecializedCyclicMma(
        BlueprintStrategy<(), SpecializedAlgorithm<AsyncPartialCyclicLoading<ColMajorTilingOrder>>>,
    ),
    SpecializedStridedCmma(BlueprintStrategy<(), SpecializedAlgorithm<AsyncPartialStridedLoading>>),
    SpecializedStridedMma(BlueprintStrategy<(), SpecializedAlgorithm<AsyncPartialStridedLoading>>),
    SpecializedTmaCmma(BlueprintStrategy<(), SpecializedAlgorithm>),
    SpecializedTmaMma(BlueprintStrategy<(), SpecializedAlgorithm>),
    OrderedDoubleCmma(BlueprintStrategy<(), OrderedDoubleBufferingAlgorithm>),
    OrderedDoubleMma(BlueprintStrategy<(), OrderedDoubleBufferingAlgorithm>),
    SimpleUnit(BlueprintStrategy<(), SimpleUnitAlgorithm>),
    DoubleUnit(BlueprintStrategy<(), DoubleUnitAlgorithm>),
    SimpleVecMat(BlueprintStrategy<(), VecMatInnerProductAlgorithm>),
    DoubleVecMat(BlueprintStrategy<(), DoubleVecMatInnerProductAlgorithm>),
    GemvUnitPerpendicular(BlueprintStrategy<(), GemvUnitPerpendicularRoutine>),
    GemvPlaneParallel(BlueprintStrategy<(), GemvPlaneParallelRoutine>),
    Naive,
    #[default]
    Auto,
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::SimpleCyclicCmma(s) => write!(f, "matmul_simple_cyclic_cmma{}", s),
            Strategy::SimpleCyclicMma(s) => write!(f, "matmul_simple_cyclic_mma{}", s),
            Strategy::SimpleStridedCmma(s) => write!(f, "matmul_simple_strided_cmma{}", s),
            Strategy::SimpleStridedMma(s) => write!(f, "matmul_simple_strided_mma{}", s),
            Strategy::SimpleTilewiseCmma(s) => write!(f, "matmul_simple_tilewise_cmma{}", s),
            Strategy::SimpleTilewiseMma(s) => write!(f, "matmul_simple_tilewise_mma{}", s),
            Strategy::SimpleAsyncStridedCmma(s) => {
                write!(f, "matmul_simple_async_strided_cmma{}", s)
            }
            Strategy::SimpleAsyncStridedMma(s) => write!(f, "matmul_simple_async_strided_mma{}", s),
            Strategy::SimpleAsyncCyclicCmma(s) => write!(f, "matmul_simple_async_cyclic_cmma{}", s),
            Strategy::SimpleAsyncCyclicMma(s) => write!(f, "matmul_simple_async_cyclic_mma{}", s),
            Strategy::SimpleTmaCmma(s) => write!(f, "matmul_simple_tma_cmma{}", s),
            Strategy::SimpleTmaMma(s) => write!(f, "matmul_simple_tma_mma{}", s),
            Strategy::DoubleCyclicCmma(s) => write!(f, "matmul_double_cyclic_cmma{}", s),
            Strategy::DoubleCyclicMma(s) => write!(f, "matmul_double_cyclic_mma{}", s),
            Strategy::DoubleTilewiseCmma(s) => write!(f, "matmul_double_tilewise_cmma{}", s),
            Strategy::DoubleTilewiseMma(s) => write!(f, "matmul_double_tilewise_mma{}", s),
            Strategy::DoubleHybridCmma(s) => write!(f, "matmul_double_hybrid_cmma{}", s),
            Strategy::DoubleHybridMma(s) => write!(f, "matmul_double_hybrid_mma{}", s),
            Strategy::DoubleAsyncCyclicCmma(s) => {
                write!(f, "matmul_double_async_cyclic_cmma{}", s)
            }
            Strategy::DoubleAsyncCyclicMma(s) => write!(f, "matmul_double_async_cyclic_mma{}", s),
            Strategy::DoubleAsyncStridedCmma(s) => {
                write!(f, "matmul_double_async_strided_cmma{}", s)
            }
            Strategy::DoubleAsyncStridedMma(s) => write!(f, "matmul_double_async_strided_mma{}", s),
            Strategy::DoubleTmaCmma(s) => write!(f, "matmul_double_tma_cmma{}", s),
            Strategy::DoubleTmaMma(s) => write!(f, "matmul_double_tma_mma{}", s),
            Strategy::SpecializedCyclicCmma(s) => write!(f, "matmul_specialized_cyclic_cmma{}", s),
            Strategy::SpecializedCyclicMma(s) => write!(f, "matmul_specialized_cyclic_mma{}", s),
            Strategy::SpecializedStridedCmma(s) => {
                write!(f, "matmul_specialized_strided_cmma{}", s)
            }
            Strategy::SpecializedStridedMma(s) => write!(f, "matmul_specialized_strided_mma{}", s),
            Strategy::SpecializedTmaCmma(s) => write!(f, "matmul_specialized_tma_cmma{}", s),
            Strategy::SpecializedTmaMma(s) => write!(f, "matmul_specialized_tma_mma{}", s),
            Strategy::OrderedDoubleCmma(s) => write!(f, "matmul_ordered_double_cmma{}", s),
            Strategy::OrderedDoubleMma(s) => write!(f, "matmul_ordered_double_mma{}", s),
            Strategy::SimpleUnit(s) => write!(f, "matmul_simple_unit{}", s),
            Strategy::DoubleUnit(s) => write!(f, "matmul_double_unit{}", s),
            Strategy::SimpleVecMat(s) => write!(f, "matmul_simple_vecmat{}", s),
            Strategy::DoubleVecMat(s) => write!(f, "matmul_double_vecmat{}", s),
            Strategy::Naive => f.write_str("matmul_naive"),
            Strategy::Auto => f.write_str("matmul_auto"),
            Strategy::GemvUnitPerpendicular(s) => write!(f, "vecmat_unit_perpendicular{}", s),
            Strategy::GemvPlaneParallel(s) => write!(f, "vecmat_plane_parallel{}", s),
        }
    }
}

#[allow(clippy::result_large_err)]
impl Strategy {
    pub(crate) fn launch_ref<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs: InputBinding<R>,
        rhs: InputBinding<R>,
        out: TensorBinding<R>,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        use TileMatmul::{Cmma, Mma};
        match self {
            Strategy::SimpleCyclicCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_simple),
                dtypes,
            ),
            Strategy::SimpleCyclicMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_simple),
                dtypes,
            ),
            Strategy::SimpleStridedCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_simple),
                dtypes,
            ),
            Strategy::SimpleStridedMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_simple),
                dtypes,
            ),
            Strategy::SimpleTilewiseCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_simple),
                dtypes,
            ),
            Strategy::SimpleTilewiseMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_simple),
                dtypes,
            ),
            Strategy::SimpleAsyncStridedCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_simple),
                dtypes,
            ),
            Strategy::SimpleAsyncStridedMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_simple),
                dtypes,
            ),
            Strategy::SimpleAsyncCyclicCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_simple),
                dtypes,
            ),
            Strategy::SimpleAsyncCyclicMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_simple),
                dtypes,
            ),
            Strategy::SimpleTmaCmma(s) => launch_tiling::launch_ref_tma(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_simple),
                dtypes,
            ),
            Strategy::SimpleTmaMma(s) => launch_tiling::launch_ref_tma(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_simple),
                dtypes,
            ),
            Strategy::DoubleCyclicCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_double),
                dtypes,
            ),
            Strategy::DoubleCyclicMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_double),
                dtypes,
            ),
            Strategy::DoubleTilewiseCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_double),
                dtypes,
            ),
            Strategy::DoubleTilewiseMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_double),
                dtypes,
            ),
            Strategy::DoubleHybridCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_double),
                dtypes,
            ),
            Strategy::DoubleHybridMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_double),
                dtypes,
            ),
            Strategy::DoubleAsyncCyclicCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_double),
                dtypes,
            ),
            Strategy::DoubleAsyncCyclicMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_double),
                dtypes,
            ),
            Strategy::DoubleAsyncStridedCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_double),
                dtypes,
            ),
            Strategy::DoubleAsyncStridedMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_double),
                dtypes,
            ),
            Strategy::DoubleTmaCmma(s) => launch_tiling::launch_ref_tma(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_double),
                dtypes,
            ),
            Strategy::DoubleTmaMma(s) => launch_tiling::launch_ref_tma(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_double),
                dtypes,
            ),
            Strategy::SpecializedCyclicCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_specialized),
                dtypes,
            ),
            Strategy::SpecializedCyclicMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_specialized),
                dtypes,
            ),
            Strategy::SpecializedStridedCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_specialized),
                dtypes,
            ),
            Strategy::SpecializedStridedMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_specialized),
                dtypes,
            ),
            Strategy::SpecializedTmaCmma(s) => launch_tiling::launch_ref_tma(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_specialized),
                dtypes,
            ),
            Strategy::SpecializedTmaMma(s) => launch_tiling::launch_ref_tma(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_specialized),
                dtypes,
            ),
            Strategy::OrderedDoubleCmma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Cmma, set_ordered),
                dtypes,
            ),
            Strategy::OrderedDoubleMma(s) => launch_tiling::launch_ref(
                client,
                lhs,
                rhs,
                out,
                &stamp_kind(s, Mma, set_ordered),
                dtypes,
            ),
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
            Strategy::GemvUnitPerpendicular(blueprint_strategy) => {
                launch_vecmat_unit_perpendicular::launch_ref(
                    client,
                    lhs,
                    rhs,
                    out,
                    blueprint_strategy,
                    dtypes,
                )
            }
            Strategy::GemvPlaneParallel(blueprint_strategy) => {
                launch_vecmat_plane_parallel::launch_ref(
                    client,
                    lhs,
                    rhs,
                    out,
                    blueprint_strategy,
                    dtypes,
                )
            }
        }
    }
}

fn auto<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &mut MatmulElems,
) -> Result<(), MatmulSetupError> {
    if let Err(err) = Strategy::SimpleCyclicCmma(Default::default()).launch_ref(
        client,
        lhs.clone(),
        rhs.clone(),
        out.clone(),
        dtypes,
    ) {
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
