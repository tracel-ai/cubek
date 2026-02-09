use crate::{launch::RuntimeConfig, routines::Routine};
use std::fmt::Display;

pub enum BlueprintStrategy<RC: RuntimeConfig, A: Routine<RC>> {
    /// Use a predefined blueprint
    Forced(A::Blueprint),
    /// Allows to give limited blueprint information, and the rest is inferred from it
    Inferred(A::Strategy),
}

impl<RC: RuntimeConfig, A: Routine<RC>> BlueprintStrategy<RC, A> {
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

impl<RC: RuntimeConfig, A: Routine<RC>> Default for BlueprintStrategy<RC, A> {
    fn default() -> Self {
        Self::Inferred(Default::default())
    }
}

impl<RC: RuntimeConfig, A: Routine<RC>> Display for BlueprintStrategy<RC, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Forced(_) => f.write_str("_forced"),
            Self::Inferred(strategy) => write!(f, "{}", strategy),
        }
    }
}

impl<RC: RuntimeConfig, A: Routine<RC>> Clone for BlueprintStrategy<RC, A> {
    fn clone(&self) -> Self {
        match self {
            Self::Forced(blueprint) => Self::Forced(blueprint.clone()),
            Self::Inferred(strategy) => Self::Inferred(strategy.clone()),
        }
    }
}
