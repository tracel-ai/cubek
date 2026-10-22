use cubek_microkernel::attention::AttentionVisibility;
use serde::{Deserialize, Serialize};

/// How a query scores the keys: which of them it sees, and the scale its dot products take
/// before the softmax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionScoring {
    pub visibility: AttentionVisibility,
    pub scale: ScoreScale,
}

impl AttentionScoring {
    /// `visibility` at the standard scale of a head of `head_dim` features, `1 / sqrt(head_dim)`.
    pub fn standard(visibility: AttentionVisibility, head_dim: usize) -> Self {
        Self {
            visibility,
            scale: ScoreScale::standard(head_dim),
        }
    }
}

/// The scale an attention's scores take before the softmax: the standard `1 / sqrt(head_dim)`
/// or a custom scale.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScoreScale {
    /// `1 / sqrt(head_dim)`.
    Standard(f32),
    /// A custom scale set by the model.
    Custom(f32),
}

impl ScoreScale {
    /// The standard scale of a head of `head_dim` features, `1 / sqrt(head_dim)`.
    pub fn standard(head_dim: usize) -> Self {
        Self::Standard(1.0 / (head_dim as f32).sqrt())
    }

    /// The scale, standard or custom.
    pub fn value(self) -> f32 {
        match self {
            Self::Standard(scale) | Self::Custom(scale) => scale,
        }
    }

    /// Which of the two the scale is, and its bits: what equality and hashing compare.
    fn identity(self) -> (core::mem::Discriminant<Self>, u32) {
        (core::mem::discriminant(&self), self.value().to_bits())
    }
}

impl PartialEq for ScoreScale {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for ScoreScale {}

impl core::hash::Hash for ScoreScale {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
    }
}
