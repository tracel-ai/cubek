use super::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A rule runs the first plan that serves and launches nothing after it: a later plan in the
/// list is a fallback, never a second launch.
#[test]
fn a_list_runs_the_first_plan_that_serves_and_nothing_after() {
    let after = Routine::serving("after", |_| true);
    let list = InOrder::new(vec![
        Routine::serving("declines", |_| false).boxed(),
        Routine::serving("serves", |_| true).boxed(),
        after.clone().boxed(),
    ]);
    assert_eq!(list.launch(&1), Ok("serves".to_string()));
    assert_eq!(after.launches(), 0);
}

/// A plan that can tell only by launching, as the multi-level selection does, still hands the list
/// on to the next plan when it declines: a decline is an answer, whenever it comes.
#[test]
fn a_decline_at_launch_hands_the_list_on() {
    let list = InOrder::new(vec![
        Routine::serving("odd inputs only", |inputs| inputs % 2 == 1).boxed(),
        Routine::serving("anything", |_| true).boxed(),
    ]);
    assert_eq!(list.launch(&1), Ok("odd inputs only".to_string()));
    assert_eq!(list.launch(&2), Ok("anything".to_string()));
}

/// A pin is a list of one: where its plan declines, the list declines with that plan's reason, so
/// the caller that named it hears why.
#[test]
fn a_pin_declines_with_its_plans_reason() {
    let pin = InOrder::new(vec![Routine::serving("pinned", |_| false).boxed()]);
    assert_eq!(
        pin.launch(&1),
        Err(Declined {
            plan: "pinned".to_string(),
            why: "declines 1".to_string(),
        })
    );
}

/// A step hands the plan after it what it made of the inputs, and its decline stops the plan from
/// running at all.
#[test]
fn a_step_hands_its_plan_what_it_made_and_its_decline_stops_it() {
    let after = Routine::serving("even inputs only", |inputs| inputs % 2 == 0);
    let doubled = Then::new(Doubled, after.clone().boxed());
    assert_eq!(doubled.launch(&3), Ok("even inputs only".to_string()));
    assert_eq!(
        doubled.launch(&0),
        Err(Declined {
            plan: "doubled".to_string(),
            why: "nothing to double".to_string(),
        })
    );
    assert_eq!(after.launches(), 1);
}

/// Two plans that try the same routines in the same order print alike, however their lists are
/// nested: a list held in a list prints flat, and a list of one as its plan.
#[test]
fn plans_that_try_the_same_routines_print_alike() {
    let nested = InOrder::new(vec![
        Routine::serving("a", |_| true).boxed(),
        Box::new(InOrder::new(vec![
            Routine::serving("b", |_| true).boxed(),
            Box::new(InOrder::new(vec![Routine::serving("c", |_| true).boxed()])),
        ])),
        Box::new(Then::new(
            Doubled,
            Box::new(InOrder::new(vec![Box::new(Fastest::new(
                "table",
                |_: &u32| "fastest".to_string(),
            ))])),
        )),
    ]);
    assert_eq!(
        nested.tree().to_string(),
        "in-order[a, b, c, doubled(fastest(table))]"
    );
}

/// A routine that serves what `serves` says, counting its launches.
#[derive(Clone)]
struct Routine {
    name: &'static str,
    serves: fn(&u32) -> bool,
    launched: Arc<AtomicUsize>,
}

impl Routine {
    fn serving(name: &'static str, serves: fn(&u32) -> bool) -> Self {
        Self {
            name,
            serves,
            launched: Arc::default(),
        }
    }

    fn boxed(self) -> BoxedPlan<u32, String> {
        Box::new(self)
    }

    fn launches(&self) -> usize {
        self.launched.load(Ordering::Relaxed)
    }
}

impl Plan<u32> for Routine {
    type Output = String;

    fn launch(&self, inputs: &u32) -> Result<String, Declined> {
        if !(self.serves)(inputs) {
            return Err(Declined::by(self, format!("declines {inputs}")));
        }
        self.launched.fetch_add(1, Ordering::Relaxed);
        Ok(self.name.to_string())
    }

    fn tree(&self) -> PlanTree {
        PlanTree::Routine(self.name.to_string())
    }
}

/// The inputs doubled, declining zero.
struct Doubled;

impl Step<u32> for Doubled {
    fn apply(&self, inputs: &u32) -> Result<u32, Declined> {
        match inputs {
            0 => Err(Declined {
                plan: "doubled".to_string(),
                why: "nothing to double".to_string(),
            }),
            n => Ok(n * 2),
        }
    }

    fn name(&self) -> String {
        "doubled".to_string()
    }
}
