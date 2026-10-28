use super::{Declined, Plan, PlanTree};

/// An autotune table as a plan: the fastest of its candidates for the inputs' key, measured the
/// first time a key is met and remembered after.
///
/// The table itself, its tuner and its candidates, is the family's: this plan names it and runs
/// it. A table serves whatever reaches it, its floor being a candidate that serves any problem,
/// so it never declines.
pub struct Fastest<I, O> {
    table: &'static str,
    launch: Box<dyn Fn(&I) -> O + Send + Sync>,
}

impl<I, O> Fastest<I, O> {
    /// The table named `table`, which `launch` runs on a call's inputs.
    pub fn new(table: &'static str, launch: impl Fn(&I) -> O + Send + Sync + 'static) -> Self {
        Self {
            table,
            launch: Box::new(launch),
        }
    }
}

impl<I, O> Plan<I> for Fastest<I, O> {
    type Output = O;

    fn launch(&self, inputs: &I) -> Result<O, Declined> {
        Ok((self.launch)(inputs))
    }

    fn tree(&self) -> PlanTree {
        PlanTree::Fastest(self.table.to_string())
    }
}
