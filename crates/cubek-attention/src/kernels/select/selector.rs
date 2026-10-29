//! Which kernel an [`AttentionOpStrategy`] selects for a call: the family's one selector, pure so
//! its rules are unit-tested with no GPU.
//!
//! The selector runs first. What it answers is how the call reads its cache, and then how the
//! kernel is chosen: by measurement on one of two autotunes, by rule, or by the caller's pin.
//! Each routine then chooses its own setting, the autotune's candidates being its settings.

use super::problem::{KvKind, QueryPositions};
use super::{AttentionKernel, AttentionOpStrategy};
use crate::kernels::plan::{
    AttentionAutotuneGroups, AttentionPlan, CacheRead, KernelLead, Rebuilt, fastest,
};
use crate::kernels::{CubeAttentionKernel, HypercubeAttentionKernel};
use cubecl::tile::launch::Refusal;
use cubek_std::plan::{InOrder, Then};

/// The kernel a rule runs at one query position, over a float cache or a packed one.
const ONE_POSITION_RULE: AttentionKernel =
    AttentionKernel::Hypercube(HypercubeAttentionKernel::LastArrivalShuffle);

/// A pin of a kernel that reads no packed cache, over a packed cache.
const PACKED_PIN_UNREAD: Refusal = Refusal::Declined {
    why: "of the microkernel attention routines, only the hypercube shuffles read a packed cache",
};

/// How a call of `queries` positions over a `cache` runs under `strategy`: the plan its launch
/// tries, the autotunes it names shipping their routines' group at `groups`.
///
/// A measuring strategy measures: a call of one query position on the one-position autotune, any
/// other on the many-positions autotune. A rule runs the hypercube shuffle merged on the last arrival
/// at one position, and the cube fragment routine elsewhere, then what each kernel's scope names
/// where it declines, down to the unit routine. A pin runs its kernel alone.
///
/// A packed cache is decoded inline at one position under every policy, by the hypercube shuffle
/// by rule, and rebuilt as floats first elsewhere. Of the pins, only the two hypercube shuffles'
/// read one.
///
/// # Errors
///
/// A pin of a kernel that reads no packed cache, over a packed cache. The launch panics on it: a
/// pin is a contract with the caller, not a preference.
pub fn select(
    strategy: AttentionOpStrategy,
    queries: QueryPositions,
    cache: KvKind,
    groups: AttentionAutotuneGroups,
) -> Result<AttentionPlan, Refusal> {
    let choice = Choice::of(strategy, queries);
    let float = || choice.plan(queries, CacheRead::Float, groups);
    match (cache, choice) {
        (KvKind::Float, _) => Ok(float()),
        (KvKind::Packed, Choice::Pin(kernel)) if kernel.reads_packed() => {
            Ok(Box::new(KernelLead::new(kernel, CacheRead::Packed)))
        }
        (KvKind::Packed, Choice::Pin(_)) => Err(PACKED_PIN_UNREAD),
        (KvKind::Packed, _) if queries == QueryPositions::One => Ok(Box::new(InOrder::new(vec![
            Choice::Rule(ONE_POSITION_RULE).plan(queries, CacheRead::Packed, groups),
            rebuilt(Choice::Rule(AttentionKernel::Unit).plan(queries, CacheRead::Float, groups)),
        ]))),
        (KvKind::Packed, _) => Ok(rebuilt(float())),
    }
}

/// How a kernel is chosen for a call: measured, by rule, or as the caller pinned it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    /// The fastest candidate of the autotune of the call's query positions.
    Measured,
    /// The kernel's lead by rule, then what its scope names where it declines.
    Rule(AttentionKernel),
    /// The kernel's lead alone: the caller asked for that kernel.
    Pin(AttentionKernel),
}

impl Choice {
    /// How `strategy` chooses for a call of `queries` positions.
    fn of(strategy: AttentionOpStrategy, queries: QueryPositions) -> Self {
        match strategy {
            AttentionOpStrategy::Specific(kernel) => Self::Pin(kernel),
            strategy if strategy.measures() => Self::Measured,
            _ => Self::Rule(match queries {
                QueryPositions::One => ONE_POSITION_RULE,
                QueryPositions::Many => AttentionKernel::Cube(CubeAttentionKernel::Fragment),
            }),
        }
    }

    /// The plan this choice runs over the cache `cache` says.
    fn plan(
        self,
        queries: QueryPositions,
        cache: CacheRead,
        groups: AttentionAutotuneGroups,
    ) -> AttentionPlan {
        let lead = |kernel| -> AttentionPlan { Box::new(KernelLead::new(kernel, cache)) };
        match self {
            Self::Measured => fastest(queries, groups.of(queries)),
            Self::Pin(kernel) => lead(kernel),
            Self::Rule(kernel) => Box::new(InOrder::new(
                core::iter::successors(Some(kernel), AttentionKernel::instead_by_rule)
                    .map(lead)
                    .collect(),
            )),
        }
    }
}

/// The packed cache rebuilt as floats, then `plan` over it.
fn rebuilt(plan: AttentionPlan) -> AttentionPlan {
    Box::new(Then::new(Rebuilt, plan))
}

#[cfg(test)]
mod tests {
    use crate::kernels::plan::AttentionAutotuneGroups;
    use crate::kernels::select::selector::*;
    use crate::kernels::*;
    use AttentionKernelHeuristic as H;
    use AttentionOpStrategy::{Autotune, Heuristic, Specific};
    use QueryPositions::{Many, One};
    use cubek_std::plan::Plan;

    const KINDS: [KvKind; 2] = [KvKind::Float, KvKind::Packed];
    const LAST_ARRIVAL_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::LastArrivalShuffle);
    const SECOND_LAUNCH_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::SecondLaunchShuffle);
    const CUBE_FRAGMENT: AttentionKernel = AttentionKernel::Cube(CubeAttentionKernel::Fragment);

    /// The tile ceiling: over a float cache the hypercube by rule at one query position and the
    /// cube elsewhere; over a packed one the hypercube decoding inline at one position, the unit
    /// routine after it, and the cache rebuilt before the cube elsewhere. Nothing measured
    /// anywhere.
    #[test]
    fn the_tile_ceiling_rules() {
        let tile = Heuristic(H::Tile);
        assert_eq!(
            plan(tile, One, KvKind::Float),
            Ok(rule(LAST_ARRIVAL_SHUFFLE, FLOAT))
        );
        assert_eq!(
            plan(tile, Many, KvKind::Float),
            Ok(rule(CUBE_FRAGMENT, FLOAT)),
            "many query positions are never measured below the burn ceiling"
        );
        assert_eq!(plan(tile, One, KvKind::Packed), Ok(packed_by_rule()));
        assert_eq!(
            plan(tile, Many, KvKind::Packed),
            Ok(rebuilt(rule(CUBE_FRAGMENT, FLOAT)))
        );
    }

    /// Where a kernel a rule named declines, the cube's plane shuffles run after its plane
    /// fragments, the hypercube shuffle merged by a second launch after the one merged on the last
    /// arrival, and the unit routine after any other: every kernel reaches the unit routine,
    /// and nothing follows it.
    #[test]
    fn every_rule_ends_at_the_unit_routine() {
        assert_eq!(
            CUBE_FRAGMENT.instead_by_rule(),
            Some(AttentionKernel::Cube(CubeAttentionKernel::Shuffle))
        );
        assert_eq!(
            LAST_ARRIVAL_SHUFFLE.instead_by_rule(),
            Some(SECOND_LAUNCH_SHUFFLE)
        );
        for kernel in AttentionKernel::all() {
            let last = std::iter::successors(Some(kernel), AttentionKernel::instead_by_rule).last();
            assert_eq!(last, Some(AttentionKernel::Unit), "{kernel:?}");
        }
        assert_eq!(AttentionKernel::Unit.instead_by_rule(), None);
    }

    /// `Autotune` measures every float call, on the autotune of its query positions; a packed call
    /// of one position has one kernel and nothing to race, and a packed call of many is rebuilt
    /// and measured as a float one.
    #[test]
    fn autotune_measures_every_float_call() {
        assert_eq!(
            plan(Autotune, One, KvKind::Float),
            Ok("fastest(decode)".into())
        );
        assert_eq!(
            plan(Autotune, Many, KvKind::Float),
            Ok("fastest(prefill)".into())
        );
        assert_eq!(plan(Autotune, One, KvKind::Packed), Ok(packed_by_rule()));
        assert_eq!(
            plan(Autotune, Many, KvKind::Packed),
            Ok(rebuilt("fastest(prefill)".into()))
        );
    }

    /// Every candidate of the family is measured, so the `Cubek` ceiling and `Autotune` name the
    /// same set.
    #[test]
    fn the_cubek_ceiling_and_autotune_agree_while_every_candidate_is_measured() {
        for cache in KINDS {
            for queries in [One, Many] {
                assert_eq!(
                    plan(Heuristic(H::Cubek), queries, cache),
                    plan(Autotune, queries, cache),
                    "{queries:?} over {cache:?}"
                );
            }
        }
    }

    /// A routine's pin runs that routine at any query length over a float cache; over a packed
    /// one only the two hypercube shuffles' run.
    #[test]
    fn a_routine_pin_runs_its_routine() {
        for kernel in AttentionKernel::all() {
            for queries in [One, Many] {
                assert_eq!(
                    plan(Specific(kernel), queries, KvKind::Float),
                    Ok(lead(kernel, FLOAT))
                );
            }
            let packed = plan(Specific(kernel), One, KvKind::Packed);
            match kernel {
                LAST_ARRIVAL_SHUFFLE | SECOND_LAUNCH_SHUFFLE => {
                    assert_eq!(packed, Ok(lead(kernel, PACKED)))
                }
                _ => assert!(packed.is_err(), "{kernel:?}"),
            }
        }
    }

    /// How a kernel's lead reads a float cache, and a packed one as it lies.
    const FLOAT: &str = "";
    const PACKED: &str = "(packed cache)";

    /// What `strategy`'s plan for a call of `queries` positions over `cache` tries, as its tree
    /// prints it.
    fn plan(
        strategy: AttentionOpStrategy,
        queries: QueryPositions,
        cache: KvKind,
    ) -> Result<String, cubecl::tile::launch::Refusal> {
        select(strategy, queries, cache, AttentionAutotuneGroups::default())
            .map(|plan| plan.tree().to_string())
    }

    /// `kernel` at its lead, reading the cache as `cache` says.
    fn lead(kernel: AttentionKernel, cache: &str) -> String {
        format!("{}@lead{cache}", kernel.name())
    }

    /// `kernel` by rule: its lead, then each one its scope names instead, down to the unit routine.
    fn rule(kernel: AttentionKernel, cache: &str) -> String {
        let leads: Vec<String> =
            std::iter::successors(Some(kernel), AttentionKernel::instead_by_rule)
                .map(|kernel| lead(kernel, cache))
                .collect();
        format!("in-order[{}]", leads.join(", "))
    }

    /// The packed cache rebuilt as floats, then `plan` over it.
    fn rebuilt(plan: String) -> String {
        format!("rebuilt({plan})")
    }

    /// What a packed cache of one query position reads under every policy: the hypercube shuffle
    /// merged on the last arrival by rule, then the unit routine over the rebuilt cache.
    fn packed_by_rule() -> String {
        let read = rule(LAST_ARRIVAL_SHUFFLE, PACKED);
        let read = read.trim_start_matches("in-order[").trim_end_matches(']');
        format!(
            "in-order[{read}, {}]",
            rebuilt(lead(AttentionKernel::Unit, FLOAT))
        )
    }
}
