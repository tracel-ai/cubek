use super::AttentionCall;
use crate::kernels::AttentionKernel;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_std::plan::{Declined, Plan, PlanTree};

/// One kernel at its lead, the first setting it lists, over the cache a call holds: a float one,
/// or a packed one as it lies through the caller's read.
pub struct KernelLead {
    kernel: AttentionKernel,
    cache: CacheRead,
}

/// Which cache a [`KernelLead`] reads: what its plan says, so a call holding the other declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheRead {
    Float,
    Packed,
}

/// A call over the other cache than the plan was built for.
const OTHER_CACHE: Refusal = Refusal::Declined {
    why: "the call holds the other cache than this kernel's plan reads",
};

impl KernelLead {
    /// `kernel` at its lead over the cache `cache` says.
    pub fn new(kernel: AttentionKernel, cache: CacheRead) -> Self {
        Self { kernel, cache }
    }
}

impl Plan<AttentionCall> for KernelLead {
    type Output = TensorHandle;

    fn launch(&self, call: &AttentionCall) -> Result<TensorHandle, Declined> {
        let launched = match (self.cache, call) {
            (CacheRead::Float, AttentionCall::Float(float)) => self
                .kernel
                .launch_lead_on(&float.hw, float.inputs.tensors().clone()),
            (CacheRead::Packed, AttentionCall::Packed(packed)) => (packed.read)(self.kernel),
            _ => Err(OTHER_CACHE),
        };
        launched.map_err(|why| Declined::by(self, why))
    }

    fn tree(&self) -> PlanTree {
        let cache = match self.cache {
            CacheRead::Float => "",
            CacheRead::Packed => "(packed cache)",
        };
        PlanTree::Routine(format!("{}@lead{cache}", self.kernel.name()))
    }
}
