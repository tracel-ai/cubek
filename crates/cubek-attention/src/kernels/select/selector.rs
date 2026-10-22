//! Which kernel an [`AttentionOpStrategy`] selects for a call: the family's one selector, pure so
//! its rules are unit-tested with no GPU.
//!
//! The selector runs first. What it answers is how the call reads its cache, and then how the
//! kernel is chosen: by measurement on one of two autotune tables, by rule, or by the caller's pin.
//! Each routine then chooses its own setting, the table's candidates being its settings.

use super::{AttentionKernel, AttentionOpStrategy};
use crate::kernels::{AttentionTensors, CubeAttentionKernel, HypercubeAttentionKernel};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// How many query positions a call has: what the rules and the tables tell calls apart by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryPositions {
    One,
    Many,
}

impl QueryPositions {
    /// The positions of a call of `seq_q` queries.
    pub fn of(seq_q: usize) -> Self {
        match seq_q {
            1 => Self::One,
            _ => Self::Many,
        }
    }
}

/// What the cache a call reads is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvKind {
    Float,
    /// TurboQuant, with or without the keys' QJL fix-up.
    Packed,
}

/// How a call reads its cache, and the kernel it reads it with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionSelection {
    /// A float cache, read by the kernel `choice` gives.
    Float(AttentionChoice),
    /// A packed cache rebuilt as floats, the QJL fix-up folded into the keys, then read by the
    /// kernel `choice` gives.
    DequantizeThen(AttentionChoice),
    /// A packed cache read as it lies, decoded by the hypercube shuffle `read` gives. Where it
    /// declines, the cache is rebuilt as floats and read by the kernel `otherwise` gives; with no
    /// `otherwise`, a pin, the launch panics.
    Packed {
        read: KernelChoice,
        otherwise: Option<AttentionChoice>,
    },
}

/// How a float attention's kernel is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionChoice {
    /// The fastest candidate of the autotune table of calls of these query positions.
    Autotune(QueryPositions),
    /// A kernel named by rule or by the caller.
    Kernel(KernelChoice),
}

/// A kernel a call runs without measuring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelChoice {
    /// The kernel's lead by rule ([`AttentionKernel::launch_by_rule_on`]).
    Rule(AttentionKernel),
    /// The kernel's lead, with nothing after it where it declines: the caller asked for that
    /// kernel.
    Pin(AttentionKernel),
}

impl KernelChoice {
    /// The chosen kernel's lead on `tensors`, by rule or as pinned.
    ///
    /// # Errors
    ///
    /// Why a pin declines, or why every kernel a rule reaches declines.
    pub fn launch_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
    ) -> Result<TensorHandle, Refusal> {
        match self {
            Self::Rule(kernel) => kernel.launch_by_rule_on(hw, tensors),
            Self::Pin(kernel) => kernel.launch_lead_on(hw, tensors),
        }
    }
}

/// The kernel a rule runs at one query position, over a float cache or a packed one.
const ONE_POSITION_RULE: AttentionKernel =
    AttentionKernel::Hypercube(HypercubeAttentionKernel::LastArrivalShuffle);

/// How a call of `queries` positions over a `cache` runs under `strategy`.
///
/// A measuring strategy measures: a call of one query position on the one-position table, any
/// other on the many-positions table. A rule runs the hypercube shuffle merged on the last arrival
/// at one position, and the cube fragment routine elsewhere. A pin runs its kernel.
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
) -> Result<AttentionSelection, Refusal> {
    let choice = match strategy {
        AttentionOpStrategy::Specific(kernel) => AttentionChoice::Kernel(KernelChoice::Pin(kernel)),
        strategy if strategy.measures() => AttentionChoice::Autotune(queries),
        _ => AttentionChoice::Kernel(KernelChoice::Rule(match queries {
            QueryPositions::One => ONE_POSITION_RULE,
            QueryPositions::Many => AttentionKernel::Cube(CubeAttentionKernel::Fragment),
        })),
    };
    match (cache, choice) {
        (KvKind::Float, choice) => Ok(AttentionSelection::Float(choice)),
        (KvKind::Packed, AttentionChoice::Kernel(KernelChoice::Pin(kernel)))
            if kernel.reads_packed() =>
        {
            Ok(AttentionSelection::Packed {
                read: KernelChoice::Pin(kernel),
                otherwise: None,
            })
        }
        (KvKind::Packed, AttentionChoice::Kernel(KernelChoice::Pin(_))) => Err(Refusal::Declined {
            why: "of the microkernel attention routines, only the hypercube shuffles read \
                          a packed cache",
        }),
        (KvKind::Packed, _) if queries == QueryPositions::One => Ok(AttentionSelection::Packed {
            read: KernelChoice::Rule(ONE_POSITION_RULE),
            otherwise: Some(AttentionChoice::Kernel(KernelChoice::Rule(
                AttentionKernel::Unit,
            ))),
        }),
        (KvKind::Packed, choice) => Ok(AttentionSelection::DequantizeThen(choice)),
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::select::selector::*;
    use crate::kernels::*;
    use AttentionChoice::{Autotune as Measured, Kernel};
    use AttentionKernelHeuristic as H;
    use AttentionOpStrategy::{Autotune, Heuristic, Specific};
    use AttentionSelection::{DequantizeThen, Float, Packed};
    use KernelChoice::{Pin, Rule};
    use QueryPositions::{Many, One};

    const KINDS: [KvKind; 2] = [KvKind::Float, KvKind::Packed];
    const LAST_ARRIVAL_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::LastArrivalShuffle);
    const SECOND_LAUNCH_SHUFFLE: AttentionKernel =
        AttentionKernel::Hypercube(HypercubeAttentionKernel::SecondLaunchShuffle);
    const CUBE_FRAGMENT: AttentionKernel = AttentionKernel::Cube(CubeAttentionKernel::Fragment);

    /// What a packed cache of one query position reads under every policy: the hypercube shuffle
    /// merged on the last arrival by rule, then the unit routine over the rebuilt cache.
    const PACKED_BY_RULE: AttentionSelection = Packed {
        read: Rule(LAST_ARRIVAL_SHUFFLE),
        otherwise: Some(Kernel(Rule(AttentionKernel::Unit))),
    };

    /// The tile ceiling: over a float cache the hypercube by rule at one query position and the
    /// cube elsewhere; over a packed one the hypercube decoding inline at one position, the unit
    /// routine after it, and the cache rebuilt before the cube elsewhere. Nothing measured
    /// anywhere.
    #[test]
    fn the_tile_ceiling_rules() {
        let tile = Heuristic(H::Tile);
        assert_eq!(
            select(tile, One, KvKind::Float),
            Ok(Float(Kernel(Rule(LAST_ARRIVAL_SHUFFLE))))
        );
        assert_eq!(
            select(tile, Many, KvKind::Float),
            Ok(Float(Kernel(Rule(CUBE_FRAGMENT)))),
            "many query positions are never measured below the burn ceiling"
        );
        assert_eq!(select(tile, One, KvKind::Packed), Ok(PACKED_BY_RULE));
        assert_eq!(
            select(tile, Many, KvKind::Packed),
            Ok(DequantizeThen(Kernel(Rule(CUBE_FRAGMENT))))
        );
    }

    /// Where a kernel a rule named declines, the cube's plane unit groups run after its plane
    /// fragments, the hypercube shuffle merged by a second launch after the one merged on the last
    /// arrival, and the unit routine after any other: every kernel reaches the unit routine,
    /// and nothing follows it.
    #[test]
    fn every_rule_ends_at_the_unit_routine() {
        assert_eq!(
            CUBE_FRAGMENT.instead_by_rule(),
            Some(AttentionKernel::Cube(CubeAttentionKernel::UnitGroup))
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

    /// `Autotune` measures every float call, on the table of its query positions; a packed call
    /// of one position has one kernel and nothing to race, and a packed call of many is rebuilt
    /// and measured as a float one.
    #[test]
    fn autotune_measures_every_float_call() {
        for queries in [One, Many] {
            assert_eq!(
                select(Autotune, queries, KvKind::Float),
                Ok(Float(Measured(queries)))
            );
        }
        assert_eq!(select(Autotune, One, KvKind::Packed), Ok(PACKED_BY_RULE));
        assert_eq!(
            select(Autotune, Many, KvKind::Packed),
            Ok(DequantizeThen(Measured(Many)))
        );
    }

    /// Every candidate of the family is on a table, so the cubek ceiling and `Autotune` name the
    /// same set.
    #[test]
    fn the_cubek_ceiling_and_autotune_agree_while_every_candidate_is_on_a_table() {
        for cache in KINDS {
            for queries in [One, Many] {
                assert_eq!(
                    select(Heuristic(H::Cubek), queries, cache),
                    select(Autotune, queries, cache),
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
                    select(Specific(kernel), queries, KvKind::Float),
                    Ok(Float(Kernel(Pin(kernel))))
                );
            }
            let packed = select(Specific(kernel), One, KvKind::Packed);
            match kernel {
                LAST_ARRIVAL_SHUFFLE | SECOND_LAUNCH_SHUFFLE => assert_eq!(
                    packed,
                    Ok(Packed {
                        read: Pin(kernel),
                        otherwise: None
                    })
                ),
                _ => assert!(packed.is_err(), "{kernel:?}"),
            }
        }
    }
}
