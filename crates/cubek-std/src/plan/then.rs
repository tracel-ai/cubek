use super::{BoxedPlan, Declined, Plan, PlanTree};

/// What a call's inputs go through before a plan reads them, such as packed inputs decoded into
/// floats.
pub trait Step<I> {
    /// What the plan reads instead of `inputs`.
    ///
    /// # Errors
    ///
    /// Why the step declines `inputs`.
    fn apply(&self, inputs: &I) -> Result<I, Declined>;

    /// What [`PlanTree`] calls the step: `decoded`.
    fn name(&self) -> String;
}

/// A step, then a plan over what the step hands it.
///
/// A step that ran is not undone where the plan after it declines: a list holding this one then
/// tries its next plan on the inputs as they were, and the step's work is spent.
pub struct Then<I, O> {
    step: Box<dyn Step<I> + Send + Sync>,
    plan: BoxedPlan<I, O>,
}

impl<I, O> Then<I, O> {
    /// `plan` over what `step` hands it.
    pub fn new(step: impl Step<I> + Send + Sync + 'static, plan: BoxedPlan<I, O>) -> Self {
        Self {
            step: Box::new(step),
            plan,
        }
    }
}

impl<I, O> Plan<I> for Then<I, O> {
    type Output = O;

    /// The step on `inputs`, then the plan on what it hands back.
    ///
    /// # Errors
    ///
    /// The step's decline, or the plan's.
    fn launch(&self, inputs: &I) -> Result<O, Declined> {
        self.plan.launch(&self.step.apply(inputs)?)
    }

    fn tree(&self) -> PlanTree {
        PlanTree::Then {
            step: self.step.name(),
            plan: Box::new(self.plan.tree()),
        }
    }
}
