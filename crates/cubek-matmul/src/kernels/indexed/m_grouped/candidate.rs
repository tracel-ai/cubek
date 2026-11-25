//! A dense routine at one of the settings listed for a grouped product
//! ([`MGroupedMatmulCandidate`]), as a caller that measures the grouped routines names and
//! launches it.

use crate::kernels::indexed::m_grouped::{MGroupedMatmulTensors, MGroupedSettings, PaddedRows};
use crate::kernels::{ListedFor, Listing, MatmulRoutine, Product};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// A dense routine at one grouped setting, and the kinds of stack the setting is listed for: one
/// entry of the grouped product's lists ([`MGroupedMatmulCandidates`](super::MGroupedMatmulCandidates)).
#[derive(Debug, Clone)]
pub struct MGroupedMatmulCandidate<R: Routine> {
    routine: R,
    listing: Listing<MGroupedSettings<R::Settings>>,
}

impl<R: Routine> MGroupedMatmulCandidate<R> {
    /// The setting this candidate launches the routine at, which names it.
    pub fn settings(&self) -> &MGroupedSettings<R::Settings> {
        &self.listing.settings
    }

    /// The grouped product `tensors` states, by this candidate's routine at its setting, on the
    /// device the operands live on, handed back padded to the setting's chunks.
    ///
    /// # Assumptions
    ///
    /// The product has a row at least.
    ///
    /// # Errors
    ///
    /// A stack of a kind the setting is not listed for, declined before compiling anything, and
    /// what [`MGroupedMatmulTensors::launch`] declines, each named with the setting.
    ///
    /// # Panics
    ///
    /// On no row, which breaks the assumption.
    pub fn launch<Mk, SM>(&self, tensors: &MGroupedMatmulTensors) -> Result<PaddedRows, String>
    where
        Mk: MatmulMicrokernel,
        R: MatmulRoutine<Mk, SM>,
    {
        let settings = self.settings();
        self.listing
            .kind
            .admits(Product::of([tensors.stack_kind()]))
            .map_err(|decline| format!("{settings} {decline:?}"))?;
        let hw = Hardware::from_client(tensors.client());
        tensors
            .launch(&hw, &self.routine, settings)
            .map_err(|reason| format!("{settings} does not serve this problem: {reason:?}"))
    }
}

impl<R: Routine + Copy> MGroupedMatmulCandidate<R> {
    /// `routine` at each setting of `listings`, in their order.
    pub(crate) fn listed(
        routine: R,
        listings: impl IntoIterator<Item = Listing<MGroupedSettings<R::Settings>>>,
    ) -> Vec<Self> {
        listings
            .into_iter()
            .map(|listing| Self { routine, listing })
            .collect()
    }

    /// `routine` at each of `settings`, in their order, each listed for either kind of stack.
    pub(crate) fn listed_for_either(
        routine: R,
        settings: impl IntoIterator<Item = MGroupedSettings<R::Settings>>,
    ) -> Vec<Self> {
        let listings = settings.into_iter().map(|settings| Listing {
            settings,
            kind: ListedFor::Either,
        });
        Self::listed(routine, listings)
    }
}
