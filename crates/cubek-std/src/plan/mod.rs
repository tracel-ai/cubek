//! How a call runs, as a tree of plans: a routine, a step and then a plan, a list of plans tried in
//! order, or an autotune table. Each is a [`Plan`], so each nests in the others.
//!
//! A family's selector returns a plan and the caller launches it: what is tried, and what runs
//! where something declines, is the plan's, never the caller's. A caller adds what the family
//! cannot reach, a [`Step`] of its own or a plan of its own, by composing it with the family's.

mod base;
mod fastest;
mod in_order;
mod then;

pub use base::*;
pub use fastest::*;
pub use in_order::*;
pub use then::*;

#[cfg(test)]
mod tests;
