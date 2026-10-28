use core::fmt::{self, Display};

/// One way to run a call on inputs `I`.
///
/// A plan that declines has launched nothing its caller depends on, so whatever holds it may try
/// another: a routine declines before compiling, while it prepares. What a plan holds is told by
/// [`tree`](Self::tree), so a test can compare two plans by what they try.
pub trait Plan<I> {
    /// What a launch hands back.
    type Output;

    /// Run the call on `inputs`.
    ///
    /// # Errors
    ///
    /// Why the plan declines `inputs`.
    fn launch(&self, inputs: &I) -> Result<Self::Output, Declined>;

    /// What this plan tries.
    fn tree(&self) -> PlanTree;
}

/// A plan held behind a box, so plans of different types sit in one list.
pub type BoxedPlan<I, O> = Box<dyn Plan<I, Output = O> + Send + Sync>;

impl<I, O> Plan<I> for BoxedPlan<I, O> {
    type Output = O;

    fn launch(&self, inputs: &I) -> Result<O, Declined> {
        self.as_ref().launch(inputs)
    }

    fn tree(&self) -> PlanTree {
        self.as_ref().tree()
    }
}

/// Why a plan ran nothing: the plan that declined last, and its reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declined {
    /// What the declining plan tries, as [`PlanTree`] writes it.
    pub plan: String,
    /// Its reason.
    pub why: String,
}

impl Declined {
    /// `plan` declining for `why`.
    pub fn by<I>(plan: &impl Plan<I>, why: impl Display) -> Self {
        Self {
            plan: plan.tree().to_string(),
            why: why.to_string(),
        }
    }
}

impl Display for Declined {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} declines: {}", self.plan, self.why)
    }
}

impl std::error::Error for Declined {}

/// What a plan tries, as data: the shape [`Plan::tree`] reports.
///
/// It prints on one line. A list held in a list prints flat and a list of one prints as its piece,
/// so two plans that try the same routines in the same order print alike.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanTree {
    /// A routine, named with what it reads and at which setting: `plane-shuffle@default`.
    Routine(String),
    /// A step before a plan: `decoded(..)`.
    Then { step: String, plan: Box<PlanTree> },
    /// Plans tried in order: `in-order[..]`.
    InOrder(Vec<PlanTree>),
    /// An autotune table: `fastest(few-rows)`.
    Fastest(String),
}

impl PlanTree {
    /// The plans this tree tries in order, a list held in a list taken flat.
    fn in_order(&self) -> Vec<&PlanTree> {
        match self {
            Self::InOrder(plans) => plans.iter().flat_map(Self::in_order).collect(),
            other => vec![other],
        }
    }
}

impl Display for PlanTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Routine(name) => f.write_str(name),
            Self::Then { step, plan } => write!(f, "{step}({plan})"),
            Self::Fastest(table) => write!(f, "fastest({table})"),
            Self::InOrder(_) => write_in_order(f, &self.in_order()),
        }
    }
}

/// `plans` written as a list tried in order, or as its one plan.
fn write_in_order(f: &mut fmt::Formatter<'_>, plans: &[&PlanTree]) -> fmt::Result {
    if let [one] = plans {
        return write!(f, "{one}");
    }
    f.write_str("in-order[")?;
    for (i, plan) in plans.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{plan}")?;
    }
    f.write_str("]")
}
