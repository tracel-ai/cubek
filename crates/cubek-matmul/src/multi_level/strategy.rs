//! Strategies built on the batch/global/stage/tile levels.

use std::fmt::Display;

use cubecl::{client::Client, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::{
    definition::{MatmulElems, MatmulSetupError},
    multi_level::{
        components::{
            global::read::async_partial_cyclic::AsyncPartialCyclicLoading, tile::TileMatmulKind,
        },
        routines::{
            batch::{
                double_buffering::{CyclicDoubleBufferingAlgorithm, DoubleBufferingArgs},
                double_unit::DoubleUnitAlgorithm,
                gemv_innerproduct::{
                    DoubleVecMatInnerProductAlgorithm, VecMatInnerProductAlgorithm,
                },
                launch as launch_tiling,
                ordered_double_buffering::{OrderedDoubleBufferingAlgorithm, OrderedSelectionArgs},
                simple::{SimpleAlgorithm, SimpleArgs, SimpleTmaAlgorithm},
                simple_unit::SimpleUnitAlgorithm,
                specialized::{SpecializedAlgorithm, SpecializedStrategy},
            },
            gemm::{GemmRoutine, launch as launch_gemm},
            gemv_unit_perpendicular::{
                GemvUnitPerpendicularRoutine, launch as launch_gemv_unit_perpendicular,
            },
            naive::launch as launch_naive,
        },
        tile::ColMajorTilingOrder,
    },
    routine::BlueprintStrategy,
};

/// Returns a clone of `sel` with `args.tile_matmul` overridden to `kind` when
/// in Inferred mode. Forced mode is left untouched since the user-supplied
/// blueprint already carries a `tile_matmul`.
fn stamp_kind<A>(
    sel: &BlueprintStrategy<A>,
    kind: TileMatmulKind,
    set: impl FnOnce(&mut A::Strategy, TileMatmulKind),
) -> BlueprintStrategy<A>
where
    A: cubek_std::legacy::Routine<Strategy: Clone, Blueprint: Clone>,
{
    let mut sel = sel.clone();
    if let BlueprintStrategy::Inferred(args) = &mut sel {
        set(args, kind);
    }
    sel
}

fn set_simple(args: &mut SimpleArgs, kind: TileMatmulKind) {
    args.tile_matmul = kind;
}
fn set_double(args: &mut DoubleBufferingArgs, kind: TileMatmulKind) {
    args.tile_matmul = kind;
}
fn set_ordered(args: &mut OrderedSelectionArgs, kind: TileMatmulKind) {
    args.tile_matmul = kind;
}
fn set_specialized(args: &mut SpecializedStrategy, kind: TileMatmulKind) {
    args.tile_matmul = kind;
}

#[derive(Clone)]
pub enum Strategy {
    SimpleCyclicCmma(BlueprintStrategy<SimpleAlgorithm>),
    SimpleCyclicMma(BlueprintStrategy<SimpleAlgorithm>),
    SimpleTmaCmma(BlueprintStrategy<SimpleTmaAlgorithm>),
    SimpleTmaMma(BlueprintStrategy<SimpleTmaAlgorithm>),
    DoubleCyclicCmma(BlueprintStrategy<CyclicDoubleBufferingAlgorithm>),
    DoubleCyclicMma(BlueprintStrategy<CyclicDoubleBufferingAlgorithm>),
    SpecializedCyclicCmma(
        BlueprintStrategy<SpecializedAlgorithm<AsyncPartialCyclicLoading<ColMajorTilingOrder>>>,
    ),
    SpecializedCyclicMma(
        BlueprintStrategy<SpecializedAlgorithm<AsyncPartialCyclicLoading<ColMajorTilingOrder>>>,
    ),
    SpecializedTmaCmma(BlueprintStrategy<SpecializedAlgorithm>),
    SpecializedTmaMma(BlueprintStrategy<SpecializedAlgorithm>),
    OrderedDoubleCmma(BlueprintStrategy<OrderedDoubleBufferingAlgorithm>),
    OrderedDoubleMma(BlueprintStrategy<OrderedDoubleBufferingAlgorithm>),
    SimpleUnit(BlueprintStrategy<SimpleUnitAlgorithm>),
    DoubleUnit(BlueprintStrategy<DoubleUnitAlgorithm>),
    SimpleVecMat(BlueprintStrategy<VecMatInnerProductAlgorithm>),
    DoubleVecMat(BlueprintStrategy<DoubleVecMatInnerProductAlgorithm>),
    GemvUnitPerpendicular(BlueprintStrategy<GemvUnitPerpendicularRoutine>),
    Gemm(BlueprintStrategy<GemmRoutine>),
    Naive,
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::SimpleCyclicCmma(s) => write!(f, "matmul_simple_cyclic_cmma{}", s),
            Strategy::SimpleCyclicMma(s) => write!(f, "matmul_simple_cyclic_mma{}", s),
            Strategy::SimpleTmaCmma(s) => write!(f, "matmul_simple_tma_cmma{}", s),
            Strategy::SimpleTmaMma(s) => write!(f, "matmul_simple_tma_mma{}", s),
            Strategy::DoubleCyclicCmma(s) => write!(f, "matmul_double_cyclic_cmma{}", s),
            Strategy::DoubleCyclicMma(s) => write!(f, "matmul_double_cyclic_mma{}", s),
            Strategy::SpecializedCyclicCmma(s) => write!(f, "matmul_specialized_cyclic_cmma{}", s),
            Strategy::SpecializedCyclicMma(s) => write!(f, "matmul_specialized_cyclic_mma{}", s),
            Strategy::SpecializedTmaCmma(s) => write!(f, "matmul_specialized_tma_cmma{}", s),
            Strategy::SpecializedTmaMma(s) => write!(f, "matmul_specialized_tma_mma{}", s),
            Strategy::OrderedDoubleCmma(s) => write!(f, "matmul_ordered_double_cmma{}", s),
            Strategy::OrderedDoubleMma(s) => write!(f, "matmul_ordered_double_mma{}", s),
            Strategy::SimpleUnit(s) => write!(f, "matmul_simple_unit{}", s),
            Strategy::DoubleUnit(s) => write!(f, "matmul_double_unit{}", s),
            Strategy::SimpleVecMat(s) => write!(f, "matmul_simple_vecmat{}", s),
            Strategy::DoubleVecMat(s) => write!(f, "matmul_double_vecmat{}", s),
            Strategy::Naive => f.write_str("matmul_naive"),
            Strategy::GemvUnitPerpendicular(s) => write!(f, "vecmat_unit_perpendicular{}", s),
            Strategy::Gemm(s) => write!(f, "gemm{}", s),
        }
    }
}

#[allow(clippy::result_large_err)]
impl Strategy {
    pub(crate) fn launch_ref(
        &self,
        client: &Client,
        lhs: InputBinding,
        rhs: InputBinding,
        out: TensorBinding,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        use TileMatmulKind::{Cmma, Mma};
        // Every routine here reads rows off its operands' strides, and would read a tiled
        // buffer's tiles as rows; a storage-tiled operand rides the tiled cmma routine.
        for (name, binding) in [("lhs", lhs.data()), ("rhs", rhs.data())] {
            if binding.tiling.is_tiled() {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "{self}: {name} is storage-tiled, which the multi-level routines read as \
                     rows; it rides the tiled cmma routine"
                ))));
            }
        }
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
            // Ordered loading carries a hard divisibility constraint on the
            // *device's* plane size (each loading plane must own a whole
            // number of vectors per unit), so a selection recorded on one
            // device (or under an older selector) can be invalid on
            // another. The entry stays reliable by degrading to the universal
            // unit kernel on an invalid config, like the gemv/gemm routines
            // below: availability and validation errors surface unchanged.
            Strategy::OrderedDoubleCmma(s) => {
                match launch_tiling::launch_ref(
                    client,
                    lhs.clone(),
                    rhs.clone(),
                    out.clone(),
                    &stamp_kind(s, Cmma, set_ordered),
                    dtypes,
                ) {
                    Err(MatmulSetupError::InvalidConfig(_)) => {
                        Strategy::SimpleUnit(Default::default())
                            .launch_ref(client, lhs, rhs, out, dtypes)
                    }
                    other => other,
                }
            }
            Strategy::OrderedDoubleMma(s) => {
                match launch_tiling::launch_ref(
                    client,
                    lhs.clone(),
                    rhs.clone(),
                    out.clone(),
                    &stamp_kind(s, Mma, set_ordered),
                    dtypes,
                ) {
                    Err(MatmulSetupError::InvalidConfig(_)) => {
                        Strategy::SimpleUnit(Default::default())
                            .launch_ref(client, lhs, rhs, out, dtypes)
                    }
                    other => other,
                }
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
            // These two routines carry hard legality constraints (dimensions
            // must be multiples of the plane size) while autotune caches their
            // selection per *anchored* key: a winner picked on an aligned
            // representative must still run on the bucket's other raw shapes.
            // The entry stays reliable by degrading to the universal unit
            // kernel on an invalid config, like [`auto`] does: availability
            // and validation errors surface unchanged.
            Strategy::GemvUnitPerpendicular(blueprint_strategy) => {
                match launch_gemv_unit_perpendicular::launch_ref(
                    client,
                    lhs.clone(),
                    rhs.clone(),
                    out.clone(),
                    blueprint_strategy,
                    dtypes,
                ) {
                    Err(MatmulSetupError::InvalidConfig(_)) => {
                        Strategy::SimpleUnit(Default::default())
                            .launch_ref(client, lhs, rhs, out, dtypes)
                    }
                    other => other,
                }
            }
            Strategy::Gemm(blueprint_strategy) => {
                match launch_gemm::launch_ref(
                    client,
                    lhs.clone(),
                    rhs.clone(),
                    out.clone(),
                    blueprint_strategy,
                    dtypes,
                ) {
                    Err(MatmulSetupError::InvalidConfig(_)) => {
                        Strategy::SimpleUnit(Default::default())
                            .launch_ref(client, lhs, rhs, out, dtypes)
                    }
                    other => other,
                }
            }
        }
    }
}
