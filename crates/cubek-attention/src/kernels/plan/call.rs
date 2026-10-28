use crate::kernels::AttentionKernel;
use crate::kernels::tune::AttentionTuneInputs;
use cubecl::Device;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::plan::{BoxedPlan, Declined, Step};
use std::sync::Arc;

/// A plan of the attention family: what a call's launch tries, on one [`AttentionCall`].
pub type AttentionPlan = BoxedPlan<AttentionCall, TensorHandle>;

/// One attention a plan runs: over a float cache, or over a packed one as it lies.
#[derive(Clone)]
pub enum AttentionCall {
    /// Boxed: a float call holds every tensor its autotune launches on, a packed one two handles.
    Float(Box<FloatAttentionCall>),
    Packed(PackedAttentionCall),
}

/// An attention over a float cache: the tensors every kernel and autotune candidate launches on,
/// and the device it runs on.
#[derive(Clone)]
pub struct FloatAttentionCall {
    pub hw: Hardware,
    pub device: Device,
    pub inputs: AttentionTuneInputs,
}

/// An attention over a packed cache, as the caller holds it.
///
/// The cache's format is the caller's: how a kernel reads it, the query turned into its basis and
/// the output turned back, and how it is rebuilt as floats. So the caller binds both to its cache
/// here, and the plan asks for them by kernel.
#[derive(Clone)]
pub struct PackedAttentionCall {
    pub read: PackedCacheRead,
    pub rebuild: CacheRebuild,
}

/// The attention of a packed cache read as it lies by one kernel at its lead.
pub type PackedCacheRead =
    Arc<dyn Fn(AttentionKernel) -> Result<TensorHandle, Refusal> + Send + Sync>;

/// The packed cache rebuilt as floats: the float call that reads it.
pub type CacheRebuild = Arc<dyn Fn() -> Result<FloatAttentionCall, Refusal> + Send + Sync>;

/// The packed cache rebuilt as floats before a plan over the float call: a float call passes
/// through as it is.
pub struct Rebuilt;

impl Step<AttentionCall> for Rebuilt {
    fn apply(&self, call: &AttentionCall) -> Result<AttentionCall, Declined> {
        match call {
            AttentionCall::Float(_) => Ok(call.clone()),
            AttentionCall::Packed(packed) => (packed.rebuild)()
                .map(|float| AttentionCall::Float(Box::new(float)))
                .map_err(|why| Declined {
                    plan: self.name(),
                    why: why.to_string(),
                }),
        }
    }

    fn name(&self) -> String {
        "rebuilt".to_string()
    }
}
