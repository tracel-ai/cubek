//! What attention's selector is asked about: how many query positions a call has, and what the
//! cache it reads is.

/// How many query positions a call has: what the rules and autotune tell calls apart by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryPositions {
    One,
    Many,
}

impl QueryPositions {
    /// The positions of a call of `seq_q` queries.
    pub fn of(seq_q: usize) -> Self {
        match seq_q {
            1 => Self::One,
            _ => Self::Many,
        }
    }
}

/// What the cache a call reads is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    Float,
    /// TurboQuant, with or without the keys' QJL fix-up.
    Packed,
}
