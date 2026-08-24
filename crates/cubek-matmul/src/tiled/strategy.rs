//! Strategies built on the tile DSL.

use std::fmt::Display;

use cubecl::{Runtime, client::ComputeClient, prelude::TensorBinding};
use cubek_std::InputBinding;

use crate::{
    definition::{MatmulElems, MatmulSetupError},
    routine::{BlueprintStrategy, into_contiguous_if_highly_permuted},
    tiled::{
        cmma::{self, CmmaRoutine},
        cpu_gemm::{self, CpuGemmRoutine, WithLayout},
    },
};

#[derive(Clone)]
pub enum Strategy {
    /// Plane-level tensor cores over a cyclic stage. The strategy's `delivery` picks
    /// strided or TMA operands (`Unavailable` without TMA).
    Cmma(BlueprintStrategy<(), CmmaRoutine>),
    /// Register-leaf contraction sized to fit a register block. Needs no accelerator.
    CpuGemm(BlueprintStrategy<(), CpuGemmRoutine>),
}

impl Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Cmma(s) => write!(f, "cmma{}", s),
            Strategy::CpuGemm(s) => write!(f, "cpu_gemm{}", s),
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
        match self {
            Strategy::Cmma(strategy) => cmma::launch_ref(
                client,
                into_contiguous_if_highly_permuted(client, lhs)?,
                into_contiguous_if_highly_permuted(client, rhs)?,
                out,
                strategy,
                dtypes,
            ),
            Strategy::CpuGemm(strategy) => cpu_gemm::launch_ref(
                client,
                WithLayout::strided_input(into_contiguous_if_highly_permuted(client, lhs)?)?,
                WithLayout::strided_input(into_contiguous_if_highly_permuted(client, rhs)?)?,
                WithLayout::strided_output(out)?,
                strategy,
                dtypes,
            ),
        }
    }
}
