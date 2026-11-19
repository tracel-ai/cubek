use cubek_test_utils::CatalogEntry;

/// Experts in a Flash-Next feed-forward layer's stack.
const EXPERTS: usize = 512;

/// Experts the router picks for each token.
const PICKS: usize = 10;

/// The model width: `gate_up`'s contraction and `down`'s output.
const D_MODEL: usize = 2560;

/// One expert's expansion: `down`'s contraction, half of `gate_up`'s output.
const EXPERT_FF: usize = 640;

/// Which product reads the stack, and for how many tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertProduct {
    /// The decode: each token's row multiplies the `picks` slabs its route names
    /// ([`routed`](crate::kernels::indexed::routed)).
    Routed { tokens: usize },
    /// The prefill: the tokens' `tokens · picks` rows, sorted by expert, each group multiplying
    /// its expert's slab ([`m_grouped`](crate::kernels::indexed::m_grouped)).
    Grouped { tokens: usize },
}

/// One product against an NVFP4 stack of `experts` slabs of `[k, n]`, laid end to end along `n`
/// as a model stores its routed experts.
#[derive(Debug, Clone, Copy)]
pub struct ExpertStackProblem {
    pub product: ExpertProduct,
    pub k: usize,
    pub n: usize,
    pub experts: usize,
    pub picks: usize,
}

impl ExpertStackProblem {
    /// The stack's rows: every slab's `n`.
    pub fn rows(&self) -> usize {
        self.experts * self.n
    }

    /// The rows the product multiplies: one per token for the routed product, one per token and
    /// pick for the grouped one.
    pub fn lhs_rows(&self) -> usize {
        match self.product {
            ExpertProduct::Routed { tokens } => tokens,
            ExpertProduct::Grouped { tokens } => tokens * self.picks,
        }
    }

    /// The grouped product's rows per expert: row `r` goes to a well-mixed expert, so the groups
    /// are as uneven as a router's.
    pub fn group_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0; self.experts];
        for row in 0..self.lhs_rows() {
            sizes[mixed(row) % self.experts] += 1;
        }
        sizes
    }

    /// Distinct slabs one call reads: the routed product's picks for each token (distinct within
    /// a token, apart across tokens as [`routes`](super::benchmark) lays them), the grouped
    /// product's non-empty groups.
    pub fn slabs_read(&self) -> usize {
        match self.product {
            ExpertProduct::Routed { tokens } => (tokens * self.picks).min(self.experts),
            ExpertProduct::Grouped { .. } => {
                self.group_sizes().iter().filter(|&&size| size > 0).count()
            }
        }
    }
}

/// A well-mixed number from `i`.
pub(super) fn mixed(i: usize) -> usize {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (x ^ (x >> 32)) as usize
}

/// Qwen3.8-Flash-Next's two expert stacks, `gate_up` `[2560, 512 · 1280]` and `down`
/// `[640, 512 · 2560]`, under a decode of one and four tokens and a prefill of 64 and 512.
pub fn problems() -> Vec<CatalogEntry<ExpertStackProblem>> {
    let products = [
        ("decode1", ExpertProduct::Routed { tokens: 1 }),
        ("decode4", ExpertProduct::Routed { tokens: 4 }),
        ("prefill64", ExpertProduct::Grouped { tokens: 64 }),
        ("prefill512", ExpertProduct::Grouped { tokens: 512 }),
    ];
    let stacks = [
        ("gate_up", D_MODEL, 2 * EXPERT_FF),
        ("down", EXPERT_FF, D_MODEL),
    ];
    products
        .into_iter()
        .flat_map(|(product_id, product)| {
            stacks.map(|(stack_id, k, n)| {
                CatalogEntry::new(
                    format!("{product_id}_{stack_id}"),
                    format!("{product_id} {stack_id} k={k} n={n} of {EXPERTS} experts"),
                    ExpertStackProblem {
                        product,
                        k,
                        n,
                        experts: EXPERTS,
                        picks: PICKS,
                    },
                )
            })
        })
        .collect()
}
