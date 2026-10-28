use super::{BoxedPlan, Declined, Plan, PlanTree};

/// Plans tried in order: the first that does not decline runs.
///
/// A rule is one, ending on a plan that serves whatever reaches it. A list of one is a pin: what
/// it holds declines, and so does the list.
pub struct InOrder<I, O> {
    plans: Vec<BoxedPlan<I, O>>,
}

/// A list that holds no plan, which declines whatever it is handed.
const NOTHING_LISTED: &str = "the list holds no plan";

impl<I, O> InOrder<I, O> {
    /// `plans`, tried in the order given.
    pub fn new(plans: Vec<BoxedPlan<I, O>>) -> Self {
        Self { plans }
    }
}

impl<I, O> Plan<I> for InOrder<I, O> {
    type Output = O;

    /// The first plan that does not decline, run on `inputs`.
    ///
    /// # Errors
    ///
    /// The last plan's decline, where every plan declines.
    fn launch(&self, inputs: &I) -> Result<O, Declined> {
        let mut declined = None;
        for plan in &self.plans {
            match plan.launch(inputs) {
                Ok(out) => return Ok(out),
                Err(why) => declined = Some(why),
            }
        }
        Err(declined.unwrap_or_else(|| Declined::by(self, NOTHING_LISTED)))
    }

    fn tree(&self) -> PlanTree {
        PlanTree::InOrder(self.plans.iter().map(Plan::tree).collect())
    }
}
