use super::{AttentionCall, AttentionPlan};
use crate::kernels::AttentionKernel;
use crate::kernels::select::problem::QueryPositions;
use crate::kernels::tune::launch_tuned;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_std::autotune::GroupPriority;
use cubek_std::plan::{Declined, Fastest, Plan, PlanTree};

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

/// The priority each autotune's routines ship at: a caller's to set, as in the matmul family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionTableGroups {
    /// The many-positions autotune's.
    pub prefill: GroupPriority,
    /// The one-position autotune's.
    pub decode: GroupPriority,
}

impl AttentionTableGroups {
    /// The priority of the autotune of calls of `positions` query positions.
    pub fn of(self, positions: QueryPositions) -> GroupPriority {
        match positions {
            QueryPositions::One => self.decode,
            QueryPositions::Many => self.prefill,
        }
    }
}

impl Default for AttentionTableGroups {
    fn default() -> Self {
        Self {
            prefill: GroupPriority::DEFAULT,
            decode: GroupPriority::DEFAULT,
        }
    }
}

/// The autotune of calls of `positions` query positions as a plan, its routines' group at
/// `routines`: the decode autotune at one position, the prefill one at many. It measures a float
/// call; a packed cache is rebuilt before it.
pub fn fastest(positions: QueryPositions, routines: GroupPriority) -> AttentionPlan {
    let table = match positions {
        QueryPositions::One => "decode",
        QueryPositions::Many => "prefill",
    };
    Box::new(Fastest::new(
        table,
        move |call: &AttentionCall| match call {
            AttentionCall::Float(float) => {
                launch_tuned(positions, &float.device, float.inputs.clone(), routines)
            }
            AttentionCall::Packed(_) => {
                unreachable!(
                    "an autotune is planned over a float call, the packed cache rebuilt first"
                )
            }
        },
    ))
}
