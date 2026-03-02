use cubecl::{AutotuneKey, ir::ElemType};
use serde::{Deserialize, Serialize};

#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
/// Autotune key representative of reduce versions
pub struct AttentionAutotuneKey {
    elem_query: ElemType,
    elem_key: ElemType,
    elem_value: ElemType,
    elem_out: ElemType,

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
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        elem_query: ElemType,
        elem_key: ElemType,
        elem_value: ElemType,
        elem_out: ElemType,
        total_batches: usize,
        seq_q: usize,
        head_dim: usize,
        seq_kv: usize,
        val_dim: usize,
        attention_mask: bool,
    ) -> Self {
        AttentionAutotuneKey::new(
            elem_query,
            elem_key,
            elem_value,
            elem_out,
            total_batches,
            seq_q,
            head_dim,
            seq_kv,
            val_dim,
            attention_mask,
        )
    }
}
