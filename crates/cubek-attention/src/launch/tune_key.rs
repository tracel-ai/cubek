use cubecl::AutotuneKey;
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of reduce versions
pub struct AttentionAutotuneKey {
    #[autotune(anchor)]
    pub total_batches: usize,
    #[autotune(anchor)]
    pub seq_q: usize,
    #[autotune(anchor)]
    pub head_dim: usize,
    #[autotune(anchor)]
    pub seq_kv: usize,
    #[autotune(anchor)]
    pub val_dim: usize,
    pub attention_mask: bool,
}

impl AttentionAutotuneKey {
    pub fn generate(
        total_batches: usize,
        seq_q: usize,
        head_dim: usize,
        seq_kv: usize,
        val_dim: usize,
        attention_mask: bool,
    ) -> Self {
        AttentionAutotuneKey::new(
            total_batches,
            seq_q,
            head_dim,
            seq_kv,
            val_dim,
            attention_mask,
        )
    }
}
