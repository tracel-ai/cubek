use serde::{Deserialize, Serialize};

/// How a query's scores become the weights its values are summed under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionWeighting {
    /// `softmax(score)` along the keys: each weight against the rest of its row, the rows summing
    /// to one.
    Softmax,
    /// `max(score, 0)`: each weight on its own, nothing normalized, as a key selector scores
    /// cache entries against its heads.
    Relu,
}
