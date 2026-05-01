use cubek::attention::definition::{
    AttentionDims, AttentionGlobalTypes, AttentionOptions, AttentionProblem,
};

use crate::registry::ItemDescriptor;

/// Stable IDs. Changing one is a breaking change for any persisted history.
pub const PROBLEM_BERT: &str = "bert";
pub const PROBLEM_GPT2: &str = "gpt2";
pub const PROBLEM_LLAMA: &str = "llama";
pub const PROBLEM_LONG_CONTEXT: &str = "long_context";
pub const PROBLEM_ENCODER_DECODER: &str = "encoder_decoder";

pub fn problems() -> Vec<ItemDescriptor> {
    vec![
        ItemDescriptor {
            id: PROBLEM_BERT.to_string(),
            label: "BERT (b=8 h=12 sq=skv=128 d=64)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_GPT2.to_string(),
            label: "GPT-2 (b=4 h=12 sq=skv=1024 d=64, causal+mask)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_LLAMA.to_string(),
            label: "Llama (b=4 h=32 sq=skv=2048 d=128, causal+mask)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_LONG_CONTEXT.to_string(),
            label: "Long context (b=1 h=16 sq=skv=4096 d=128, causal+mask)".to_string(),
        },
        ItemDescriptor {
            id: PROBLEM_ENCODER_DECODER.to_string(),
            label: "Encoder-decoder (b=2 h=16 sq=512 skv=1024 d=128)".to_string(),
        },
    ]
}

pub(crate) fn problem_for(
    id: &str,
    global_dtypes: AttentionGlobalTypes,
) -> Option<AttentionProblem> {
    let causal_masked = AttentionOptions {
        causal: true,
        accumulator_precision: Default::default(),
    };
    Some(match id {
        PROBLEM_BERT => AttentionProblem {
            dims: AttentionDims {
                batch: 8,
                num_heads: 12,
                seq_q: 128,
                seq_kv: 128,
                head_dim: 64,
                val_dim: 64,
            },
            global_dtypes,
            masked: false,
            options: Default::default(),
            address_type: Default::default(),
        },
        PROBLEM_GPT2 => AttentionProblem {
            dims: AttentionDims {
                batch: 4,
                num_heads: 12,
                seq_q: 1024,
                seq_kv: 1024,
                head_dim: 64,
                val_dim: 64,
            },
            global_dtypes,
            masked: true,
            options: causal_masked,
            address_type: Default::default(),
        },
        PROBLEM_LLAMA => AttentionProblem {
            dims: AttentionDims {
                batch: 4,
                num_heads: 32,
                seq_q: 2048,
                seq_kv: 2048,
                head_dim: 128,
                val_dim: 128,
            },
            global_dtypes,
            masked: true,
            options: causal_masked,
            address_type: Default::default(),
        },
        PROBLEM_LONG_CONTEXT => AttentionProblem {
            dims: AttentionDims {
                batch: 1,
                num_heads: 16,
                seq_q: 4096,
                seq_kv: 4096,
                head_dim: 128,
                val_dim: 128,
            },
            global_dtypes,
            masked: true,
            options: causal_masked,
            address_type: Default::default(),
        },
        PROBLEM_ENCODER_DECODER => AttentionProblem {
            dims: AttentionDims {
                batch: 2,
                num_heads: 16,
                seq_q: 512,
                seq_kv: 1024,
                head_dim: 128,
                val_dim: 128,
            },
            global_dtypes,
            masked: false,
            options: AttentionOptions {
                causal: false,
                accumulator_precision: Default::default(),
            },
            address_type: Default::default(),
        },
        _ => return None,
    })
}
