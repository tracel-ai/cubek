use crate::definition::MatmulSelection;

#[derive(Debug, Clone)]
pub enum Selection<S> {
    /// Use a predefined MatmulSelection
    Forced(MatmulSelection),
    /// Allows to give limited MatmulSelection information, and the rest is inferred from it
    Inferred(S),
}

impl<S: Default + Clone> Selection<S> {
    pub fn maybe_forced_default(s: &Option<MatmulSelection>) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_default()
    }
    pub fn maybe_forced_or(s: &Option<MatmulSelection>, args: &S) -> Self {
        s.as_ref()
            .map(|s| Self::Forced(s.clone()))
            .unwrap_or_else(|| Self::Inferred(args.clone()))
    }
}

impl<S: Default> Default for Selection<S> {
    fn default() -> Self {
        Self::Inferred(Default::default())
    }
}
