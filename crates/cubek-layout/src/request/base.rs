//! The request itself: a set of [`Constraint`]s and the feasibility/preference predicates.

use cubecl::zspace::SmallVec;

use crate::MAX_AXES;

use super::{ConcreteLayout, Facet};

/// How strongly a [`Constraint`] binds.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Strength {
    /// Unmet means the strategy cannot run on this layout. Defines the feasible set.
    Required,
    /// Met is better but not necessary. Ranks layouts within the feasible set.
    Preferred,
}

/// A [`Facet`] plus how strongly it binds.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Constraint {
    pub facet: Facet,
    pub strength: Strength,
}

impl Constraint {
    pub fn required(facet: Facet) -> Self {
        Constraint {
            facet,
            strength: Strength::Required,
        }
    }

    pub fn preferred(facet: Facet) -> Self {
        Constraint {
            facet,
            strength: Strength::Preferred,
        }
    }
}

/// A strategy's layout wish: a set of [`Constraint`]s. Empty means it runs on any layout
/// (the common kernel). Intrinsic to the strategy, so it carries no extents and no
/// reference to the delivered layout.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct LayoutRequest {
    pub constraints: SmallVec<[Constraint; MAX_AXES]>,
}

impl LayoutRequest {
    pub fn new() -> Self {
        LayoutRequest::default()
    }

    pub fn with(mut self, constraint: Constraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// True when every `Required` constraint holds against `layout`. The feasible-set test
    /// Burn uses to decide whether a strategy is even a candidate for a given layout.
    pub fn feasible(&self, layout: &ConcreteLayout) -> bool {
        self.constraints
            .iter()
            .filter(|c| c.strength == Strength::Required)
            .all(|c| c.facet.holds(layout))
    }

    /// Count of satisfied `Preferred` constraints, higher is better. Only meaningful once
    /// `layout` is [`feasible`](Self::feasible).
    pub fn preference(&self, layout: &ConcreteLayout) -> usize {
        self.constraints
            .iter()
            .filter(|c| c.strength == Strength::Preferred)
            .filter(|c| c.facet.holds(layout))
            .count()
    }
}
