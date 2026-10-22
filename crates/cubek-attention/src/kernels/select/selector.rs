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
