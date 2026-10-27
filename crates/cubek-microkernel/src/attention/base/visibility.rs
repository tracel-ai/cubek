use core::num::NonZeroUsize;
use serde::{Deserialize, Serialize};

/// Which of the attended keys a query sees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionVisibility {
    /// Every attended key.
    Attended,
    /// None past its own position, the queries placed among the attended keys as `queries` says.
    Causal { queries: QueryPlacement },
    /// None past its own position, and none `window` or more before it: each query sees itself
    /// and the `window - 1` positions before it, a sliding window's local attention.
    Windowed {
        window: NonZeroUsize,
        queries: QueryPlacement,
    },
    /// The last `window` attended keys, whatever the query: a single position's window, which
    /// every head of its group sees when the group is read along the query axis.
    Recent { window: NonZeroUsize },
}

/// Where a causal attention's queries sit among the attended keys, which places its diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryPlacement {
    /// The last positions: the last query sees every attended key and each one before it one
    /// fewer, as a cache's newest tokens do.
    Last,
    /// The first positions: query `q` sees the keys up to the `q`-th.
    First,
}

impl AttentionVisibility {
    /// The visibility of a caller that states it as a flag and a window, as the attention seam
    /// does: none past a query's own position where `causal`, the queries placed among the keys
    /// as `queries` says, and within `window` keys of it where there is one — of its own
    /// position where `causal`, of the last attended key otherwise.
    pub fn new(causal: bool, window: Option<NonZeroUsize>, queries: QueryPlacement) -> Self {
        match (causal, window) {
            (false, None) => Self::Attended,
            (true, None) => Self::Causal { queries },
            (true, Some(window)) => Self::Windowed { window, queries },
            (false, Some(window)) => Self::Recent { window },
        }
    }

    /// Whether a query sees no key past its own position.
    pub fn is_causal(self) -> bool {
        self.queries().is_some()
    }

    /// Where the queries sit among the keys, where a query sees none past its own position.
    pub fn queries(self) -> Option<QueryPlacement> {
        match self {
            Self::Causal { queries } | Self::Windowed { queries, .. } => Some(queries),
            Self::Attended | Self::Recent { .. } => None,
        }
    }

    /// How many keys a window spans, where one bounds the keys a query sees from below.
    pub fn window(self) -> Option<NonZeroUsize> {
        match self {
            Self::Attended | Self::Causal { .. } => None,
            Self::Windowed { window, .. } | Self::Recent { window } => Some(window),
        }
    }

    /// This visibility for a query at the last attended position:
    /// `Causal` becomes `Attended` and `Windowed { w }` becomes `Recent { w }`.
    pub fn of_last_position(self) -> Self {
        Self::new(false, self.window(), QueryPlacement::Last)
    }
}
