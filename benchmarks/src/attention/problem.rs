use cubek::attention::definition::{
    AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem,
};

use crate::registry::CatalogEntry;

/// Client-independent description of an attention benchmark problem. The full
/// [`AttentionProblem`] also needs a [`AttentionGlobalTypes`] which depends on
/// the runtime client (for `mask_dtype`), so we build that lazily in
/// [`build_problem`] from a spec + dtypes.
#[derive(Clone)]
pub struct AttentionSpec {
    pub dims: AttentionDims,
    pub masked: bool,
    pub options: AttentionOptions,
}

pub fn build_problem(
    spec: &AttentionSpec,
    global_dtypes: AttentionGlobalTypes,
) -> AttentionProblem {
    AttentionProblem {
        dims: spec.dims.clone(),
        global_dtypes,
        masked: spec.masked,
        options: spec.options.clone(),
        address_type: Default::default(),
    }
}

fn dims(
    batch: usize,
    num_heads: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
) -> AttentionDims {
    AttentionDims {
        batch,
        num_heads,
        seq_q,
        seq_kv,
        head_dim,
        val_dim: head_dim,
    }
}

pub fn problems() -> Vec<CatalogEntry<AttentionSpec>> {
    let causal_masked = AttentionOptions {
        causal: true,
        accumulator_precision: Default::default(),
    };
    vec![
        CatalogEntry::new(
            "bert",
            "BERT (b=8 h=12 sq=skv=128 d=64)",
            AttentionSpec {
                dims: dims(8, 12, 128, 128, 64),
                masked: false,
                options: Default::default(),
            },
        ),
        CatalogEntry::new(
            "gpt2",
            "GPT-2 (b=4 h=12 sq=skv=1024 d=64, causal+mask)",
            AttentionSpec {
                dims: dims(4, 12, 1024, 1024, 64),
                masked: true,
                options: causal_masked.clone(),
            },
        ),
        CatalogEntry::new(
            "llama",
            "Llama (b=4 h=32 sq=skv=2048 d=128, causal+mask)",
            AttentionSpec {
                dims: dims(4, 32, 2048, 2048, 128),
                masked: true,
                options: causal_masked.clone(),
            },
        ),
        CatalogEntry::new(
            "long_context",
            "Long context (b=1 h=16 sq=skv=4096 d=128, causal+mask)",
            AttentionSpec {
                dims: dims(1, 16, 4096, 4096, 128),
                masked: true,
                options: causal_masked.clone(),
            },
        ),
        CatalogEntry::new(
            "encoder_decoder",
            "Encoder-decoder (b=2 h=16 sq=512 skv=1024 d=128)",
            AttentionSpec {
                dims: dims(2, 16, 512, 1024, 128),
                masked: false,
                options: AttentionOptions {
                    causal: false,
                    accumulator_precision: Default::default(),
                },
            },
        ),
        CatalogEntry::new(
            "mask_causal_4096",
            "Masked+Causal (b=1 h=4 sq=4096 skv=4096 d=64)",
            AttentionSpec {
                dims: dims(1, 4, 4096, 4096, 64),
                masked: true,
                options: AttentionOptions {
                    causal: true,
                    accumulator_precision: Default::default(),
                },
            },
        ),
    ]
}
