//! The floor under the few-rows and many-rows autotunes: the candidate that serves any product.

use super::{MatmulProduct, RoutineIdentity};
use crate::kernels::UnitRegistersMatmulRoutine;
use crate::kernels::plan::{Decoded, MatmulCall};
use crate::kernels::tune::MatmulRoutineCandidate;
use crate::kernels::tune::key::MatmulTuneKey;
use cubecl::tune::Tunable;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidateIdentity, TuneCandidate};
use cubek_std::plan::Step;

/// Both autotunes' floor: the unit routine at its default, every packed input decoded first
/// ([`Decoded`]). It asks nothing of the device, so it serves what no other candidate binds: an
/// input laid out no body can describe, or two packed inputs whose scales group `k` apart, which
/// every routine reading packed inputs as stored declines. A float input passes through as it is.
///
/// It is named apart from the unit routine, as an [`activation_decoded_identity`](crate::kernels::tune::MatmulPackedCandidate::activation_decoded_identity) is: one set of
/// candidates serves float and packed calls alike, and over a packed input the routine cannot be
/// launched again on its own, since the floor decodes first.
pub(super) fn decoded(
    hw: &Hardware,
) -> Tunable<MatmulTuneKey, MatmulCall, MatmulProduct, RoutineIdentity> {
    let lead = MatmulRoutineCandidate::list(UnitRegistersMatmulRoutine, hw).swap_remove(0);
    let identity = CandidateIdentity::Named(format!("{} decoded floor", lead.strategy()));
    Tunable::identified(identity, move |call: MatmulCall| {
        let decoded = Decoded.apply(&call).map_err(|why| why.to_string())?;
        lead.launch(decoded)
    })
}
