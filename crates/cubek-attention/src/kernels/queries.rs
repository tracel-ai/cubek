//! The queries of an attention, and where each user's sit among its keys.

use cubecl::tile::launch::Refusal;
use cubek_matmul::kernels::tensor::PlainTensor;

/// What an attention routine declines where the users' queries are packed.
pub const PACKED_QUERIES_UNREAD: Refusal = Refusal::Declined {
    why: "the attention routines read queries as [b, h, q, d], every user the same q",
};

/// The queries of an attention's users, and each user's attended length `lengths`, `u32` `[b]`
/// on the device, or `[1]` for every user: user `b` attends its first `lengths[b]` keys, and its
/// `q_b` queries are the last `q_b` of them, at positions `lengths[b] - q_b .. lengths[b]`, causal
/// from there.
///
/// A decode step is either kind with one query a user.
#[derive(Debug, Clone)]
pub enum AttentionQueries {
    /// Every user's `q` queries, `[b, h, q, d]`.
    Dense {
        query: PlainTensor,
        lengths: PlainTensor,
    },
    /// The users' queries one after another, `[Σ q_b, h, d]`: user `b`'s are
    /// `ends[b - 1] .. ends[b]` along the first dim (from `0` for the first), of the `u32`
    /// cumulative counts `ends` `[b]`. `longest` is the most queries any user has, which the
    /// launch's grid covers.
    Packed {
        query: PlainTensor,
        ends: PlainTensor,
        longest: usize,
        lengths: PlainTensor,
    },
}
