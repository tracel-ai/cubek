//! Unified `launch_ref` entry point for the convolution kernel family.
//!
//! Picks the right `Routine` impl from `ConvAlgorithm`, threads the
//! `Strategy` (`Specific` / `Forced`) into a matmul `BlueprintStrategy`, and
//! dispatches to the per-operation helper based on `ConvolutionInputs`.

use cubecl::{Runtime, client::ComputeClient};
use cubek_matmul::{
    components::tile_matmul::TileMatmulKind,
    definition::{MatmulElems, TilingBlueprint},
    routines::{BlueprintStrategy, Routine as MatmulRoutine, TilingArgs},
};

use crate::components::ConvolutionOperation;
use crate::definition::ConvBlueprint;

fn blueprint_operation(blueprint: &ConvBlueprint) -> ConvolutionOperation {
    match blueprint {
        ConvBlueprint::Forward(_) => ConvolutionOperation::Forward,
        ConvBlueprint::BackwardData(_) => ConvolutionOperation::BackwardData,
        ConvBlueprint::BackwardWeight(_) => ConvolutionOperation::BackwardWeight,
    }
}

use crate::{
    components::{ConvSetupError, global::args::RuntimeArgs},
    kernels::{backward_data, backward_weight, forward},
    launch::{
        ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy, strategy::AcceleratedTileKind,
    },
    routines::{
        Routine,
        simple::{
            SimpleAsyncCyclicConv, SimpleAsyncStridedConv, SimpleAsyncTmaConv,
            SimpleSyncCyclicConv, SimpleSyncStridedConv, SimpleSyncTilewiseConv,
        },
        specialized::{
            SpecializedAsyncCyclicConv, SpecializedAsyncStridedConv, SpecializedTmaConv,
        },
    },
};

/// Map `AcceleratedTileKind` → matmul's `TileMatmulKind`.
pub(crate) fn tile_kind_to_dispatch(kind: AcceleratedTileKind) -> TileMatmulKind {
    match kind {
        AcceleratedTileKind::Cmma => TileMatmulKind::Cmma,
        AcceleratedTileKind::Mma => TileMatmulKind::Mma,
    }
}

/// The single public convolution entry point.
///
/// Routes the `inputs` (whose discriminant is the operation) and `strategy`
/// (algorithm + tile-matmul kind, optionally a forced blueprint) into the right
/// generic `Routine` and per-operation launch helper.
#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime, const N_SPATIAL: usize>(
    strategy: &Strategy,
    client: &ComputeClient<R>,
    inputs: ConvolutionInputs<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let (algorithm, tile_kind, forced_matmul) = match strategy {
        Strategy::Inferred {
            algorithm,
            tile_kind,
        } => (*algorithm, *tile_kind, None),
        Strategy::Forced {
            algorithm,
            blueprint,
        } => {
            debug_assert_eq!(
                inputs.operation(),
                blueprint_operation(blueprint),
                "Strategy::Forced blueprint variant does not match the inputs operation",
            );
            let matmul = blueprint.matmul().clone();
            // For Forced, tile_kind is encoded inside the matmul blueprint, so
            // the explicit tile_kind here is unused; we pass Cmma as a benign
            // default (it gets overwritten by the forced blueprint).
            (*algorithm, AcceleratedTileKind::Cmma, Some(matmul))
        }
    };

    // Backward-data does not currently support the TMA reading strategy.
    if inputs.operation() == ConvolutionOperation::BackwardData
        && algorithm == ConvAlgorithm::SimpleAsyncTma
    {
        return Err(crate::kernels::backward_data::launch::unsupported_tma_error());
    }

    dispatch_routine::<R, N_SPATIAL>(
        algorithm,
        tile_kind,
        forced_matmul,
        client,
        inputs,
        args,
        dtypes,
    )
}

/// Dispatch on `ConvAlgorithm` to instantiate the right concrete `Routine`
/// generic, then forward to the per-operation helper.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn dispatch_routine<R: Runtime, const N_SPATIAL: usize>(
    algorithm: ConvAlgorithm,
    tile_kind: AcceleratedTileKind,
    forced_matmul: Option<TilingBlueprint>,
    client: &ComputeClient<R>,
    inputs: ConvolutionInputs<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError> {
    let kind = tile_kind_to_dispatch(tile_kind);
    match algorithm {
        ConvAlgorithm::SimpleSyncCyclic => dispatch_inputs::<R, N_SPATIAL, SimpleSyncCyclicConv>(
            client,
            inputs,
            args,
            kind,
            forced_matmul,
            dtypes,
        ),
        ConvAlgorithm::SimpleSyncStrided => dispatch_inputs::<R, N_SPATIAL, SimpleSyncStridedConv>(
            client,
            inputs,
            args,
            kind,
            forced_matmul,
            dtypes,
        ),
        ConvAlgorithm::SimpleSyncTilewise => {
            dispatch_inputs::<R, N_SPATIAL, SimpleSyncTilewiseConv>(
                client,
                inputs,
                args,
                kind,
                forced_matmul,
                dtypes,
            )
        }
        ConvAlgorithm::SimpleAsyncCyclic => dispatch_inputs::<R, N_SPATIAL, SimpleAsyncCyclicConv>(
            client,
            inputs,
            args,
            kind,
            forced_matmul,
            dtypes,
        ),
        ConvAlgorithm::SimpleAsyncStrided => {
            dispatch_inputs::<R, N_SPATIAL, SimpleAsyncStridedConv>(
                client,
                inputs,
                args,
                kind,
                forced_matmul,
                dtypes,
            )
        }
        ConvAlgorithm::SimpleAsyncTma => dispatch_inputs::<R, N_SPATIAL, SimpleAsyncTmaConv>(
            client,
            inputs,
            args,
            kind,
            forced_matmul,
            dtypes,
        ),
        ConvAlgorithm::SpecializedAsyncCyclic => {
            dispatch_inputs::<R, N_SPATIAL, SpecializedAsyncCyclicConv>(
                client,
                inputs,
                args,
                kind,
                forced_matmul,
                dtypes,
            )
        }
        ConvAlgorithm::SpecializedAsyncStrided => {
            dispatch_inputs::<R, N_SPATIAL, SpecializedAsyncStridedConv>(
                client,
                inputs,
                args,
                kind,
                forced_matmul,
                dtypes,
            )
        }
        ConvAlgorithm::SpecializedTma => dispatch_inputs::<R, N_SPATIAL, SpecializedTmaConv>(
            client,
            inputs,
            args,
            kind,
            forced_matmul,
            dtypes,
        ),
    }
}

/// Branch on operation and forward to the per-op launcher.
///
/// All three per-op `ConcreteArgs` traits share the same name and the same
/// blanket impls on `TensorArgs<RuntimeArgs>` / `TensorMapArgs<RuntimeArgs>`,
/// so the where clause simply requires an impl per operation.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
fn dispatch_inputs<R: Runtime, const N_SPATIAL: usize, Rt: Routine<Blueprint = TilingBlueprint>>(
    client: &ComputeClient<R>,
    inputs: ConvolutionInputs<R>,
    args: ConvolutionArgs<N_SPATIAL>,
    tile_matmul: TileMatmulKind,
    forced_matmul: Option<TilingBlueprint>,
    dtypes: MatmulElems,
) -> Result<(), ConvSetupError>
where
    Rt::Args: forward::args::ConcreteArgs<Rt::MatmulRoutine>
        + backward_data::args::ConcreteArgs<Rt::MatmulRoutine>
        + backward_weight::args::ConcreteArgs<Rt::MatmulRoutine>,
    Rt::Strategy: TilingArgs,
{
    let blueprint_strategy = build_blueprint_strategy::<Rt>(tile_matmul, forced_matmul);

    match inputs {
        ConvolutionInputs::Forward {
            input,
            weight,
            bias,
            out,
        } => forward::launch::launch_internal::<R, N_SPATIAL, Rt>(
            client,
            input,
            weight,
            bias,
            out,
            args,
            &blueprint_strategy,
            dtypes,
        ),
        ConvolutionInputs::BackwardData {
            out_grad,
            weights,
            in_grad,
        } => backward_data::launch::launch_internal::<R, N_SPATIAL, Rt>(
            client,
            out_grad,
            weights,
            in_grad,
            args,
            &blueprint_strategy,
            dtypes,
        ),
        ConvolutionInputs::BackwardWeight {
            input,
            out_grad,
            weight_grad,
        } => backward_weight::launch::launch_internal::<R, N_SPATIAL, Rt>(
            client,
            input,
            out_grad,
            weight_grad,
            args,
            &blueprint_strategy,
            dtypes,
        ),
    }
}

/// Build a matmul `BlueprintStrategy` from either a forced `TilingBlueprint`
/// (extracted from `ConvBlueprint`) or an `Inferred` strategy stamped with the
/// requested tile-matmul kind.
fn build_blueprint_strategy<Rt: Routine<Blueprint = TilingBlueprint>>(
    tile_matmul: TileMatmulKind,
    forced_matmul: Option<TilingBlueprint>,
) -> BlueprintStrategy<RuntimeArgs, Rt::MatmulRoutine>
where
    Rt::Strategy: TilingArgs,
{
    match forced_matmul {
        Some(matmul) => BlueprintStrategy::Forced(matmul),
        None => {
            let mut s = <Rt::MatmulRoutine as MatmulRoutine<RuntimeArgs>>::Strategy::default();
            s.set_tile_matmul(tile_matmul);
            BlueprintStrategy::Inferred(s)
        }
    }
}
