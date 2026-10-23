use cubek_microkernel::attention::AttentionVisibility;
use cubek_test_utils::CatalogEntry;

/// One backward pass's shape: the query's `[batch, heads, seq_q, head_dim]`, the keys' and
/// values' `[batch, kv_heads, seq_kv, head_dim]`, and which keys a query sees.
#[derive(Debug, Clone, Copy)]
pub struct AttentionBackwardSpec {
    pub batch: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub seq_q: usize,
    pub seq_kv: usize,
    pub head_dim: usize,
    pub visibility: AttentionVisibility,
}

/// Training shapes: every query sees every key, or every key up to its own position.
pub fn problems() -> Vec<CatalogEntry<AttentionBackwardSpec>> {
    let spec = |batch, heads, kv_heads, seq, head_dim, visibility| AttentionBackwardSpec {
        batch,
        heads,
        kv_heads,
        seq_q: seq,
        seq_kv: seq,
        head_dim,
        visibility,
    };
    vec![
        CatalogEntry::new(
            "bert",
            "BERT (b=8 h=12 s=128 d=64)",
            spec(8, 12, 12, 128, 64, AttentionVisibility::Attended),
        ),
        CatalogEntry::new(
            "small_causal",
            "Small causal (b=1 h=4 s=512 d=64)",
            spec(1, 4, 4, 512, 64, AttentionVisibility::Causal),
        ),
        CatalogEntry::new(
            "gpt2",
            "GPT-2 (b=4 h=12 s=1024 d=64, causal)",
            spec(4, 12, 12, 1024, 64, AttentionVisibility::Causal),
        ),
        CatalogEntry::new(
            "llama_gqa",
            "Llama GQA (b=1 h=32 kv=8 s=2048 d=128, causal)",
            spec(1, 32, 8, 2048, 128, AttentionVisibility::Causal),
        ),
    ]
}
