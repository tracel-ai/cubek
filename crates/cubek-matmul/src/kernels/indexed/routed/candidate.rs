//! A dense routine at one of the settings listed for a routed product
//! ([`RoutedMatmulCandidate`]), as a caller that measures the routed routines names and launches
//! it.

use crate::kernels::indexed::routed::RoutedMatmulTensors;
use crate::kernels::{Listing, MatmulRoutine};
use cubecl::std::tensor::TensorHandle;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// A dense routine at one setting, and the kinds of stack the setting is listed for: one entry of
/// the routed product's lists ([`RoutedMatmulCandidates`](super::RoutedMatmulCandidates)).
#[derive(Debug, Clone)]
pub struct RoutedMatmulCandidate<R: Routine> {
    routine: R,
    listing: Listing<R::Settings>,
}

impl<R: Routine> RoutedMatmulCandidate<R> {
    /// The setting this candidate launches the routine at, which names it.
    pub fn settings(&self) -> &R::Settings {
        &self.listing.settings
    }

    /// The routed product `tensors` states, by this candidate's routine at its setting, on the
    /// device the operands live on: `[rows, slots, n]` at the rows' element.
    ///
    /// # Errors
    ///
    /// A stack of a kind the setting is not listed for, declined before compiling anything, and
    /// what [`RoutedMatmulTensors::launch`] declines, each named with the setting.
    pub fn launch<Mk, SM>(&self, tensors: &RoutedMatmulTensors) -> Result<TensorHandle, String>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let settings = self.settings();
        self.listing
            .kind
            .admits(tensors.product())
            .map_err(|decline| format!("{settings} {decline:?}"))?;
        let hw = Hardware::from_client(tensors.client());
        tensors
            .launch(&hw, &self.routine, settings)
            .map_err(|reason| format!("{settings} does not serve this problem: {reason:?}"))
    }
}

impl<R: Routine + Copy> RoutedMatmulCandidate<R> {
    /// `routine` at each setting of `listings`, in their order.
    pub(crate) fn listed(
        routine: R,
        listings: impl IntoIterator<Item = Listing<R::Settings>>,
    ) -> Vec<Self> {
        listings
            .into_iter()
            .map(|listing| Self { routine, listing })
            .collect()
    }
}
