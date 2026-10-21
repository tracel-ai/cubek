use core::fmt::{Debug, Display};

/// What every routine has, whatever its problem: a blueprint it launches, and a strategy that
/// states only what a caller cares about, the routine inferring the rest.
pub trait Routine {
    type Strategy;
    type Blueprint;
}

/// How a routine gets its blueprint: forced as given, or inferred from a strategy.
pub enum BlueprintStrategy<R: Routine> {
    /// Use a predefined blueprint.
    Forced(R::Blueprint),
    /// Infer the blueprint, from the limited information the strategy gives.
    Inferred(R::Strategy),
}

impl<R: Routine<Strategy: Clone + Default, Blueprint: Clone>> BlueprintStrategy<R> {
    /// `blueprint` forced where there is one, the default strategy otherwise.
    pub fn maybe_forced_default(blueprint: &Option<R::Blueprint>) -> Self {
        blueprint
            .as_ref()
            .map(|blueprint| Self::Forced(blueprint.clone()))
            .unwrap_or_default()
    }

    /// `blueprint` forced where there is one, `strategy` otherwise.
    pub fn maybe_forced_or(blueprint: &Option<R::Blueprint>, strategy: &R::Strategy) -> Self {
        blueprint
            .as_ref()
            .map(|blueprint| Self::Forced(blueprint.clone()))
            .unwrap_or_else(|| Self::Inferred(strategy.clone()))
    }
}

impl<R: Routine<Strategy: Default>> Default for BlueprintStrategy<R> {
    fn default() -> Self {
        Self::Inferred(R::Strategy::default())
    }
}

impl<R: Routine<Strategy: Clone, Blueprint: Clone>> Clone for BlueprintStrategy<R> {
    fn clone(&self) -> Self {
        match self {
            Self::Forced(blueprint) => Self::Forced(blueprint.clone()),
            Self::Inferred(strategy) => Self::Inferred(strategy.clone()),
        }
    }
}

impl<R: Routine<Strategy: Debug, Blueprint: Debug>> Debug for BlueprintStrategy<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Forced(blueprint) => f.debug_tuple("Forced").field(blueprint).finish(),
            Self::Inferred(strategy) => f.debug_tuple("Inferred").field(strategy).finish(),
        }
    }
}

impl<R: Routine<Strategy: Display>> Display for BlueprintStrategy<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Forced(_) => f.write_str("_forced"),
            Self::Inferred(strategy) => write!(f, "{strategy}"),
        }
    }
}
