//! The problem a routine computes.

use crate::kernels::refusals::{
    RECURRENCE_WALK_BINDING_BUFFERS, RECURRENCE_WALK_BINDINGS_PAST_DEVICE,
    RECURRENCE_WALK_ELEMENT_NOT_FLOAT, RECURRENCE_WALK_ELEMENT_UNVERIFIED,
    RECURRENCE_WALK_HEADS_EMPTY, RECURRENCE_WALK_SLOTS_BINDING_BUFFERS,
};
use crate::kernels::walk::{CubeDeltaWalkRoutine, CubeDeltaWalkRoutineSettings};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Axis, Space};
use cubecl::tune::Work;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_microkernel::linear_recurrence::walk::{RecurrenceWalkOperand, RecurrenceWalkOperandMap};
use cubek_microkernel::linear_recurrence::{
    BATCH, HEAD_IN_GROUP, KEPT_STATE, KEY_FEATURE, KEY_HEAD, POSITION, RecurrenceHeads, STORE_SLOT,
    VALUE_FEATURE,
};
use cubek_std::Hardware;
use cubek_std::routine::{EstimatedProblem, Routine};
use serde::{Deserialize, Serialize};

/// The element every operand of the walk but its output is in: `f32`, as the recurrence sums a
/// decayed running state, which is where a narrow float costs accuracy.
pub const RECURRENCE_WALK_ELEMENT: ElemType = ElemType::Float(FloatKind::F32);

/// The elements the output is written at: the activations' float elements.
const OUTPUT_ELEMENTS: [ElemType; 3] = [
    ElemType::Float(FloatKind::F16),
    ElemType::Float(FloatKind::BF16),
    RECURRENCE_WALK_ELEMENT,
];

/// The operands a walk in a store's slots binds: every one but the kept states.
const SLOTS_OPERANDS: [RecurrenceWalkOperand; 7] = [
    RecurrenceWalkOperand::Query,
    RecurrenceWalkOperand::Key,
    RecurrenceWalkOperand::Value,
    RecurrenceWalkOperand::Decay,
    RecurrenceWalkOperand::Strength,
    RecurrenceWalkOperand::State,
    RecurrenceWalkOperand::Out,
];

/// Where a walk's states live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecurrenceWalkStateLayout {
    /// A state tensor the walk resumes from, one a sequence, and kept states it writes to a tensor
    /// of their own: the state after each of the last `kept` positions.
    Owned { kept: NonZeroUsize },
    /// A store of `slots` states, a table placing each sequence at its slot: the walk reads the
    /// sequence's state there and writes the state after its last position back, which is the one
    /// state a slot holds.
    Slots { slots: NonZeroUsize },
}

impl RecurrenceWalkStateLayout {
    /// States kept for a rewind: the state after each of the last this many positions.
    pub fn kept(&self) -> NonZeroUsize {
        match self {
            Self::Owned { kept } => *kept,
            Self::Slots { .. } => NonZeroUsize::MIN,
        }
    }

    /// The buffers a walk over this layout binds, the metadata buffer included.
    pub fn binding_buffers(&self) -> u32 {
        match self {
            Self::Owned { .. } => RECURRENCE_WALK_BINDING_BUFFERS,
            Self::Slots { .. } => RECURRENCE_WALK_SLOTS_BINDING_BUFFERS,
        }
    }

    /// The operands of the walk this layout binds: the kept states are an operand of the owned
    /// walk alone.
    pub fn operands(&self) -> &'static [RecurrenceWalkOperand] {
        match self {
            Self::Owned { .. } => &RecurrenceWalkOperand::ALL,
            Self::Slots { .. } => &SLOTS_OPERANDS,
        }
    }

    /// The axis a level cuts for the kept states, and its extent: the owned walk's.
    pub fn kept_axis(&self) -> Option<(Axis, usize)> {
        match self {
            Self::Owned { kept } => Some((KEPT_STATE, kept.get())),
            Self::Slots { .. } => None,
        }
    }

    /// The axis of a store's slots, and its extent: no level cuts it.
    pub fn store_axis(&self) -> Option<(Axis, usize)> {
        match self {
            Self::Owned { .. } => None,
            Self::Slots { slots } => Some((STORE_SLOT, slots.get())),
        }
    }
}

/// Where a walk's states live, as far as its kernel cares: a [`RecurrenceWalkStateLayout`] without
/// the store's size, which the kernel leaves to the launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RecurrenceWalkStateKind {
    Owned { kept: NonZeroUsize },
    Slots,
}

impl RecurrenceWalkStateKind {
    /// The layout a problem of this kind prepares on: a store of one slot, as no level cuts them.
    pub fn layout(self) -> RecurrenceWalkStateLayout {
        match self {
            Self::Owned { kept } => RecurrenceWalkStateLayout::Owned { kept },
            Self::Slots => RecurrenceWalkStateLayout::Slots {
                slots: NonZeroUsize::MIN,
            },
        }
    }
}

impl From<RecurrenceWalkStateLayout> for RecurrenceWalkStateKind {
    fn from(layout: RecurrenceWalkStateLayout) -> Self {
        match layout {
            RecurrenceWalkStateLayout::Owned { kept } => Self::Owned { kept },
            RecurrenceWalkStateLayout::Slots { .. } => Self::Slots,
        }
    }
}

/// The gated delta rule over `batch` sequences of `positions` positions, as the host describes it:
/// `key_heads · heads_per_group` value heads, each reading the queries and keys of the key head it is
/// grouped under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrenceWalkProblem {
    pub batch: usize,
    pub positions: usize,
    pub heads: RecurrenceHeads,
    /// The element the output is written at, rounded from the `f32` read: the activations'.
    pub output_element: ElemType,
    /// Where the states live, and how many are kept.
    pub state: RecurrenceWalkStateLayout,
    /// The alignment of each operand's buffer in elements, over the axes the walk names. The kept
    /// states are bound by the owned walk alone: a walk in a store's slots holds
    /// [`ElemAlignment::ONE`] for them, which nothing reads.
    pub alignments: RecurrenceWalkOperandMap<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: RecurrenceWalkOperandMap<u64>,
}

impl RecurrenceWalkProblem {
    /// Whether the walk writes its output at `output_element`: a plain float the activations are
    /// in.
    ///
    /// # Errors
    ///
    /// [`RECURRENCE_WALK_ELEMENT_NOT_FLOAT`] for an element that is not a float, a packed one
    /// included, [`RECURRENCE_WALK_ELEMENT_UNVERIFIED`] for a float the walk is not verified at.
    pub fn admit_output(output_element: ElemType) -> Result<(), Refusal> {
        match output_element {
            element if OUTPUT_ELEMENTS.contains(&element) => Ok(()),
            element if element.is_float() => Err(RECURRENCE_WALK_ELEMENT_UNVERIFIED),
            _ => Err(RECURRENCE_WALK_ELEMENT_NOT_FLOAT),
        }
    }

    /// Whether the walk reads an operand other than its output at `element`: plain `f32`.
    ///
    /// # Errors
    ///
    /// As [`admit_output`](Self::admit_output).
    pub fn admit_operand(element: ElemType) -> Result<(), Refusal> {
        match element {
            RECURRENCE_WALK_ELEMENT => Ok(()),
            element if element.is_float() => Err(RECURRENCE_WALK_ELEMENT_UNVERIFIED),
            _ => Err(RECURRENCE_WALK_ELEMENT_NOT_FLOAT),
        }
    }

    /// The first position whose state is kept.
    pub fn first_kept(&self) -> usize {
        self.positions - self.state.kept().get()
    }

    /// Whether a device launches the walk as a routine prepares it, whichever routine places it:
    /// what every routine declines a problem with before it places anything.
    ///
    /// # Errors
    ///
    /// The first of: no head ([`RECURRENCE_WALK_HEADS_EMPTY`]), more buffers bound than the device
    /// binds ([`RECURRENCE_WALK_BINDINGS_PAST_DEVICE`]).
    pub fn admit_launch(&self, hw: &Hardware) -> Result<(), Refusal> {
        let heads = &self.heads;
        if [
            heads.key_heads,
            heads.heads_per_group,
            heads.key_features,
            heads.value_features,
        ]
        .contains(&0)
        {
            return Err(RECURRENCE_WALK_HEADS_EMPTY);
        }
        if self.state.binding_buffers() > hw.properties.hardware.max_bindings {
            return Err(RECURRENCE_WALK_BINDINGS_PAST_DEVICE);
        }
        Ok(())
    }

    /// Whether a device launches the walk at all: the floor, the walk at one run a cube and the
    /// fewest registers, preparing, which includes [`admit_launch`](Self::admit_launch). Every
    /// routine's guarantee rests on it, and a head too wide for the floor is declined by the
    /// refusal that names what it overran.
    ///
    /// # Errors
    ///
    /// What the floor declines, [`admit_launch`](Self::admit_launch)'s among them.
    pub fn admit(&self, hw: &Hardware) -> Result<(), Refusal> {
        CubeDeltaWalkRoutine
            .prepare(hw, self, &CubeDeltaWalkRoutineSettings::FLOOR)
            .map(|_| ())
    }

    /// Every axis of the problem at its extent: what a routine partitions. The kept states sit
    /// outside the state's rows and columns, so each is a whole state.
    pub fn space(&self) -> Space {
        let heads = &self.heads;
        let mut axes = vec![
            (BATCH, self.batch),
            (POSITION, self.positions),
            (KEY_HEAD, heads.key_heads),
            (HEAD_IN_GROUP, heads.heads_per_group),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
        ];
        axes.extend(
            self.state
                .kept_axis()
                .into_iter()
                .chain(self.state.store_axis()),
        );
        Space::new(&axes)
    }

    /// A problem of dense operands, each starting where its buffer does: what a key describes, and
    /// what a routine's pure tests prepare against.
    pub fn dense(
        batch: usize,
        positions: usize,
        heads: RecurrenceHeads,
        output_element: ElemType,
        state: RecurrenceWalkStateLayout,
    ) -> Self {
        let rows =
            |shape: &[usize]| ElemAlignment::new(shape, &row_major_contiguous_strides(shape));
        let RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features,
            value_features,
        } = heads;
        Self {
            batch,
            positions,
            heads,
            output_element,
            state,
            alignments: RecurrenceWalkOperandMap {
                query: rows(&[batch, positions, key_heads, key_features]),
                key: rows(&[batch, positions, key_heads, key_features]),
                value: rows(&[batch, positions, key_heads, heads_per_group, value_features]),
                decay: rows(&[batch, positions, key_heads, heads_per_group]),
                strength: rows(&[batch, positions, key_heads, heads_per_group]),
                state: rows(&[
                    state.store_axis().map_or(batch, |(_, slots)| slots),
                    key_heads,
                    heads_per_group,
                    key_features,
                    value_features,
                ]),
                out: rows(&[batch, positions, key_heads, heads_per_group, value_features]),
                kept: match state {
                    RecurrenceWalkStateLayout::Owned { kept } => rows(&[
                        kept.get(),
                        batch,
                        key_heads,
                        heads_per_group,
                        key_features,
                        value_features,
                    ]),
                    RecurrenceWalkStateLayout::Slots { .. } => ElemAlignment::ONE,
                },
            },
            start_bytes: RecurrenceWalkOperandMap::splat(0),
        }
    }

    /// Bytes of one state.
    fn state_bytes(&self) -> usize {
        self.heads.key_features * self.heads.value_features * RECURRENCE_WALK_ELEMENT.size()
    }
}

impl EstimatedProblem for RecurrenceWalkProblem {
    /// Each key head's queries and keys, each head's values, scalars and outputs, the state read
    /// once and each kept state written once; per head and position, the decay, the recall, the
    /// write and the read are each a pass over the state.
    fn demand(&self) -> Work {
        let heads = &self.heads;
        let (value_heads, positions) = (self.batch * heads.value_heads(), self.positions);
        let input_bytes = RECURRENCE_WALK_ELEMENT.size();
        let keyed = self.batch * heads.key_heads * positions * 2 * heads.key_features;
        let valued = value_heads * positions * (heads.value_features + 2);
        let read = value_heads * positions * heads.value_features * self.output_element.size();
        let states = value_heads * (1 + self.state.kept().get()) * self.state_bytes();
        Work {
            compute_ops: 7 * value_heads * positions * heads.key_features * heads.value_features,
            bytes: (keyed + valued) * input_bytes + read + states,
        }
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use cubek_std::Hardware;
    use cubek_std::devices;

    /// `hw` as a device that binds as many buffers as the walk's launch does: the fixtures that
    /// bind eight are what the admission tests decline, and the routine tests would otherwise
    /// prepare on none of them.
    pub(crate) fn binding_enough(mut hw: Hardware) -> Hardware {
        devices::topology(&mut hw).max_bindings = 32;
        hw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_microkernel::linear_recurrence::walk::cube::CubeDeltaWalkMicrokernel;
    use cubek_std::devices;
    use cubek_std::microkernel::Microkernel;

    fn sixteen_heads() -> RecurrenceHeads {
        RecurrenceHeads {
            key_heads: 16,
            heads_per_group: 3,
            key_features: 128,
            value_features: 128,
        }
    }

    fn walk(heads: RecurrenceHeads) -> RecurrenceWalkProblem {
        RecurrenceWalkProblem::dense(
            1,
            8,
            heads,
            RECURRENCE_WALK_ELEMENT,
            RecurrenceWalkStateLayout::Owned {
                kept: NonZeroUsize::MIN,
            },
        )
    }

    /// The launch needs eight buffers and the metadata buffer: a device binding fewer declines it,
    /// whichever routine would place it.
    #[test]
    fn a_device_binding_fewer_buffers_than_the_launch_is_declined() {
        let problem = walk(sixteen_heads());
        for hw in [
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::strix_halo(),
        ] {
            assert_eq!(
                hw.properties.hardware.max_bindings, 8,
                "the fixture is one that binds eight"
            );
            assert_eq!(
                problem.admit(&hw),
                Err(RECURRENCE_WALK_BINDINGS_PAST_DEVICE),
                "{hw:?}"
            );
        }
        assert_eq!(problem.admit(&devices::h100()), Ok(()));
    }

    /// Eight operands and the metadata buffer: a device binding eight declines, nine admits.
    #[test]
    fn the_metadata_buffer_counts_against_the_devices_bindings() {
        let problem = walk(sixteen_heads());
        let mut hw = devices::h100();
        devices::topology(&mut hw).max_bindings = RECURRENCE_WALK_BINDING_BUFFERS - 1;
        assert_eq!(
            problem.admit(&hw),
            Err(RECURRENCE_WALK_BINDINGS_PAST_DEVICE)
        );
        devices::topology(&mut hw).max_bindings = RECURRENCE_WALK_BINDING_BUFFERS;
        assert_eq!(problem.admit(&hw), Ok(()));
    }

    fn in_slots() -> RecurrenceWalkProblem {
        let mut problem = walk(sixteen_heads());
        problem.state = RecurrenceWalkStateLayout::Slots {
            slots: NonZeroUsize::new(32).unwrap(),
        };
        problem
    }

    /// The walk in a store's slots binds nine operands and the metadata buffer: a device binding
    /// ten declines the one in a store's slots by name and admits the one over states it owns.
    #[test]
    fn the_slots_walk_binds_one_buffer_more_than_the_owned_walk() {
        let problem = in_slots();
        let mut hw = devices::h100();
        devices::topology(&mut hw).max_bindings = RECURRENCE_WALK_SLOTS_BINDING_BUFFERS - 1;
        assert_eq!(
            problem.admit(&hw),
            Err(RECURRENCE_WALK_BINDINGS_PAST_DEVICE)
        );
        assert_eq!(walk(sixteen_heads()).admit(&hw), Ok(()));
        devices::topology(&mut hw).max_bindings = RECURRENCE_WALK_SLOTS_BINDING_BUFFERS;
        assert_eq!(problem.admit(&hw), Ok(()));
        for hw in [
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::strix_halo(),
        ] {
            assert_eq!(
                problem.admit(&hw),
                Err(RECURRENCE_WALK_BINDINGS_PAST_DEVICE)
            );
        }
    }

    /// An output element that is packed or an integer is declined as not a float, a float the walk
    /// is not verified at as unverified, and the activations' floats are admitted; an operand
    /// other than `f32` is declined the same way.
    #[test]
    fn elements_that_are_not_plain_floats_are_declined() {
        use cubecl::ir::UIntKind;
        let (f16, f32) = (ElemType::Float(FloatKind::F16), RECURRENCE_WALK_ELEMENT);
        let packed = ElemType::UInt(UIntKind::U32);
        let f64 = ElemType::Float(FloatKind::F64);
        for element in [f16, ElemType::Float(FloatKind::BF16), f32] {
            assert_eq!(RecurrenceWalkProblem::admit_output(element), Ok(()));
        }
        assert_eq!(
            RecurrenceWalkProblem::admit_output(packed),
            Err(RECURRENCE_WALK_ELEMENT_NOT_FLOAT)
        );
        assert_eq!(
            RecurrenceWalkProblem::admit_output(f64),
            Err(RECURRENCE_WALK_ELEMENT_UNVERIFIED)
        );
        assert_eq!(RecurrenceWalkProblem::admit_operand(f32), Ok(()));
        assert_eq!(
            RecurrenceWalkProblem::admit_operand(f16),
            Err(RECURRENCE_WALK_ELEMENT_UNVERIFIED)
        );
        assert_eq!(
            RecurrenceWalkProblem::admit_operand(packed),
            Err(RECURRENCE_WALK_ELEMENT_NOT_FLOAT)
        );
    }

    /// Heads with no feature or no head describe no walk.
    #[test]
    fn heads_that_describe_no_walk_are_declined() {
        let hw = devices::h100();
        let heads = sixteen_heads();
        for heads in [
            RecurrenceHeads {
                key_heads: 0,
                ..heads
            },
            RecurrenceHeads {
                heads_per_group: 0,
                ..heads
            },
            RecurrenceHeads {
                key_features: 0,
                ..heads
            },
            RecurrenceHeads {
                value_features: 0,
                ..heads
            },
        ] {
            assert_eq!(
                walk(heads).admit(&hw),
                Err(RECURRENCE_WALK_HEADS_EMPTY),
                "{heads:?}"
            );
        }
    }

    /// A head wider than a cube's units hold at the fewest registers is declined by the floor's
    /// refusal, not prepared into a launch the device refuses.
    #[test]
    fn a_head_too_wide_for_the_floor_is_declined() {
        let hw = super::fixtures::binding_enough(devices::wgpu_no_subgroups());
        let wide = RecurrenceHeads {
            key_heads: 1,
            heads_per_group: 1,
            key_features: 1 << 20,
            value_features: 4,
        };
        assert_eq!(
            walk(wide).admit(&hw),
            Err(Refusal::Declined {
                why: CubeDeltaWalkMicrokernel::DECLINE
            })
        );
        assert_eq!(walk(sixteen_heads()).admit(&hw), Ok(()));
    }
}
