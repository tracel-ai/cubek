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
        tile_matmul::{cmma::CmmaMatmul, mma::MmaMatmul},
    },
    definition::{MatmulElems, MatmulSetupError},
    launch::launch_tiling,
    routines::{
        BlueprintStrategy, interleaved::InterleavedAlgorithm, simple::SimpleBarrierAlgorithm,
    },
};

type Cmma = CmmaMatmul;
type Mma = MmaMatmul;

/// Non-public strategy variants reserved for test coverage.
#[allow(clippy::type_complexity)]
#[derive(Clone)]
pub enum TestStrategy {
    SimpleBarrierCooperativeCmma(
        BlueprintStrategy<(), SimpleBarrierAlgorithm<Cmma, AsyncFullCooperativeLoading>>,
    ),
    SimpleBarrierCooperativeMma(
        BlueprintStrategy<(), SimpleBarrierAlgorithm<Mma, AsyncFullCooperativeLoading>>,
    ),
    SimpleBarrierCyclicCmma(
        BlueprintStrategy<
            (),
            SimpleBarrierAlgorithm<Cmma, AsyncFullCyclicLoading<ColMajorTilingOrder>>,
        >,
    ),
    SimpleBarrierCyclicMma(
        BlueprintStrategy<
            (),
            SimpleBarrierAlgorithm<Mma, AsyncFullCyclicLoading<ColMajorTilingOrder>>,
        >,
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
        match self {
            Self::SimpleBarrierCooperativeCmma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, sel, dtypes)
            }
            Self::SimpleBarrierCooperativeMma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, sel, dtypes)
            }
            Self::SimpleBarrierCyclicCmma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, sel, dtypes)
            }
            Self::SimpleBarrierCyclicMma(sel) => {
                launch_tiling::launch_ref(client, lhs, rhs, out, sel, dtypes)
            }
            Self::Interleaved(sel) => launch_tiling::launch_ref(client, lhs, rhs, out, sel, dtypes),
        }
    }
}
