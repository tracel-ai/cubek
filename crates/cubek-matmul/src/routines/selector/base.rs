use crate::routines::Routine;
use std::fmt::Debug;

// #[derive(Debug, Clone)]
pub enum BlueprintStrategy<A: Routine> {
    /// Use a predefined blueprint
    Forced(A::Blueprint),
    /// Allows to give limited blueprint information, and the rest is inferred from it
    Inferred(A::Strategy),
}

impl<A: Routine> BlueprintStrategy<A> {
    pub fn maybe_forced_default(s: &Option<A::Blueprint>) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_default()
    }
    pub fn maybe_forced_or(s: &Option<A::Blueprint>, args: &A::Strategy) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_else(|| Self::Inferred(args.clone()))
    }
}

impl<A: Routine> Default for BlueprintStrategy<A> {
    fn default() -> Self {
        Self::Inferred(Default::default())
    }
}

impl<A: Routine> Debug for BlueprintStrategy<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forced(arg0) => f.debug_tuple("Forced").field(arg0).finish(),
            Self::Inferred(arg0) => f.debug_tuple("Inferred").field(arg0).finish(),
        }
    }
}

impl<A: Routine> Clone for BlueprintStrategy<A> {
    fn clone(&self) -> Self {
        match self {
            Self::Forced(arg0) => Self::Forced(arg0.clone()),
            Self::Inferred(arg0) => Self::Inferred(arg0.clone()),
        }
    }
}
