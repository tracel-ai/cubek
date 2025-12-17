use crate::routines::Routine;
use std::fmt::Display;

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

impl<A: Routine> Display for BlueprintStrategy<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forced(_) => f.write_str("_forced"),
            Self::Inferred(strategy) => write!(f, "{}", strategy),
        }
    }
}

impl<A: Routine> Clone for BlueprintStrategy<A> {
    fn clone(&self) -> Self {
        match self {
            Self::Forced(blueprint) => Self::Forced(blueprint.clone()),
            Self::Inferred(strategy) => Self::Inferred(strategy.clone()),
        }
    }
}
