//! Strategy variants that are only exposed for the test suite.
//!
//! These routines are exercised by the full test tree but are not part of the
//! publicly supported [`Strategy`] enum: either they are experimental
//! or they use loading combinations that are not wired into the production selector.

use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::{
    components::{
        global::read::{
            async_full_cooperative::AsyncFullCooperativeLoading,
            async_full_cyclic::AsyncFullCyclicLoading,
        },
        stage::ColMajorTilingOrder,
        tile_matmul::TileMatmul,
    },
    definition::{MatmulElems, MatmulSetupError},
    launch::launch_tiling,
    routines::{
        BlueprintStrategy, Routine, TilingArgs, interleaved::InterleavedAlgorithm,
        simple::SimpleBarrierAlgorithm,
    },
};

/// Non-public strategy variants reserved for test coverage.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub enum TestStrategy {
    SimpleBarrierCooperativeCmma(
        BlueprintStrategy<(), SimpleBarrierAlgorithm<AsyncFullCooperativeLoading>>,
    ),
    SimpleBarrierCooperativeMma(
        BlueprintStrategy<(), SimpleBarrierAlgorithm<AsyncFullCooperativeLoading>>,
    ),
    SimpleBarrierCyclicCmma(
        BlueprintStrategy<(), SimpleBarrierAlgorithm<AsyncFullCyclicLoading<ColMajorTilingOrder>>>,
    ),
    SimpleBarrierCyclicMma(
        BlueprintStrategy<(), SimpleBarrierAlgorithm<AsyncFullCyclicLoading<ColMajorTilingOrder>>>,
    ),
    Interleaved(BlueprintStrategy<(), InterleavedAlgorithm>),
}

impl Display for TestStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SimpleBarrierCooperativeCmma(s) => {
                write!(f, "matmul_simple_barrier_cooperative_cmma{}", s)
            }
            Self::SimpleBarrierCooperativeMma(s) => {
                write!(f, "matmul_simple_barrier_cooperative_mma{}", s)
            }
            Self::SimpleBarrierCyclicCmma(s) => {
                write!(f, "matmul_simple_barrier_cyclic_cmma{}", s)
            }
            Self::SimpleBarrierCyclicMma(s) => {
                write!(f, "matmul_simple_barrier_cyclic_mma{}", s)
            }
            Self::Interleaved(s) => write!(f, "matmul_interleaved{}", s),
        }
    }
}

fn with_kind<RC, A>(sel: &BlueprintStrategy<RC, A>, kind: TileMatmul) -> BlueprintStrategy<RC, A>
where
    RC: crate::launch::RuntimeConfig,
    A: Routine<RC>,
    A::Strategy: TilingArgs,
{
    let mut sel = sel.clone();
    if let BlueprintStrategy::Inferred(args) = &mut sel {
        args.set_tile_matmul(kind);
    }
    sel
}

#[allow(clippy::result_large_err)]
impl TestStrategy {
    pub fn launch_ref<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        lhs: InputBinding<R>,
        rhs: InputBinding<R>,
        out: TensorBinding<R>,
        dtypes: &mut MatmulElems,
    ) -> Result<(), MatmulSetupError> {
        use TileMatmul::{Cmma, Mma};
        match self {
            Self::SimpleBarrierCooperativeCmma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, &with_kind(sel, Cmma), dtypes)
            }
            Self::SimpleBarrierCooperativeMma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, &with_kind(sel, Mma), dtypes)
            }
            Self::SimpleBarrierCyclicCmma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, &with_kind(sel, Cmma), dtypes)
            }
            Self::SimpleBarrierCyclicMma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, &with_kind(sel, Mma), dtypes)
            }
            Self::Interleaved(sel) => launch_tiling::launch_ref(client, lhs, rhs, out, sel, dtypes),
        }
    }
}
