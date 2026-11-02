//! The problem a routine computes.

use crate::kernels::refusals::{
    GATED_DELTA_WALK_BINDING_BUFFERS, GATED_DELTA_WALK_BINDINGS_PAST_DEVICE,
    GATED_DELTA_WALK_ELEMENT_NOT_FLOAT, GATED_DELTA_WALK_ELEMENT_UNVERIFIED,
    GATED_DELTA_WALK_HEADS_EMPTY,
};
use crate::kernels::walk::{CubeGatedDeltaWalkRoutine, CubeGatedDeltaWalkRoutineSettings};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Space;
use cubecl::tile::launch::Refusal;
use cubecl::tile::layout::ElemAlignment;
use cubecl::tune::Work;
use cubecl::zspace::striding::row_major_contiguous_strides;
use cubek_microkernel::gated_delta::walk::GatedDeltaWalkOperandMap;
use cubek_microkernel::gated_delta::{
    BATCH, GROUP, GatedDeltaHeads, KEPT_STATE, KEY_FEATURE, KEY_HEAD, POSITION, VALUE_FEATURE,
};
use cubek_std::Hardware;
use cubek_std::routine::{EstimatedProblem, Routine};

/// The element every operand of the walk but its output is in: `f32`, as the recurrence sums a
/// decayed running state, which is where a narrow float costs accuracy.
pub const GATED_DELTA_WALK_ELEMENT: ElemType = ElemType::Float(FloatKind::F32);

/// The elements the output is written at: the activations' float elements.
const OUTPUT_ELEMENTS: [ElemType; 3] = [
    ElemType::Float(FloatKind::F16),
    ElemType::Float(FloatKind::BF16),
    GATED_DELTA_WALK_ELEMENT,
];

/// The gated delta rule over `batch` sequences of `positions` positions, as the host describes it:
/// `key_heads · group` value heads, each reading the queries and keys of the key head it is
/// grouped under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatedDeltaWalkProblem {
    pub batch: usize,
    pub positions: usize,
    pub heads: GatedDeltaHeads,
    /// The element the output is written at, rounded from the `f32` read: the activations'.
    pub output_element: ElemType,
    /// States kept for a rewind: the state after each of the last this many positions.
    pub kept: NonZeroUsize,
    /// The alignment of each operand's buffer in elements, over the axes the walk names.
    pub alignments: GatedDeltaWalkOperandMap<ElemAlignment>,
    /// Bytes each operand's view starts past its buffer's own start.
    pub start_bytes: GatedDeltaWalkOperandMap<u64>,
}

impl GatedDeltaWalkProblem {
    /// Whether the walk writes its output at `output_element`: a plain float the activations are
    /// in.
    ///
    /// # Errors
    ///
    /// [`GATED_DELTA_WALK_ELEMENT_NOT_FLOAT`] for an element that is not a float, a packed one
    /// included, [`GATED_DELTA_WALK_ELEMENT_UNVERIFIED`] for a float the walk is not verified at.
    pub fn admit_output(output_element: ElemType) -> Result<(), Refusal> {
        match output_element {
            element if OUTPUT_ELEMENTS.contains(&element) => Ok(()),
            element if element.is_float() => Err(GATED_DELTA_WALK_ELEMENT_UNVERIFIED),
            _ => Err(GATED_DELTA_WALK_ELEMENT_NOT_FLOAT),
        }
    }

    /// Whether the walk reads an operand other than its output at `element`: plain `f32`.
    ///
    /// # Errors
    ///
    /// As [`admit_output`](Self::admit_output).
    pub fn admit_operand(element: ElemType) -> Result<(), Refusal> {
        match element {
            GATED_DELTA_WALK_ELEMENT => Ok(()),
            element if element.is_float() => Err(GATED_DELTA_WALK_ELEMENT_UNVERIFIED),
            _ => Err(GATED_DELTA_WALK_ELEMENT_NOT_FLOAT),
        }
    }

    /// The first position whose state is kept.
    pub fn first_kept(&self) -> usize {
        self.positions - self.kept.get()
    }

    /// Whether a device launches the walk as a routine prepares it, whichever routine places it:
    /// what every routine declines a problem with before it places anything.
    ///
    /// # Errors
    ///
    /// The first of: no head ([`GATED_DELTA_WALK_HEADS_EMPTY`]), more buffers bound than the device
    /// binds ([`GATED_DELTA_WALK_BINDINGS_PAST_DEVICE`]).
    pub fn admit_launch(&self, hw: &Hardware) -> Result<(), Refusal> {
        let heads = &self.heads;
        if [
            heads.key_heads,
            heads.group,
            heads.key_features,
            heads.value_features,
        ]
        .contains(&0)
        {
            return Err(GATED_DELTA_WALK_HEADS_EMPTY);
        }
        if GATED_DELTA_WALK_BINDING_BUFFERS > hw.properties.hardware.max_bindings {
            return Err(GATED_DELTA_WALK_BINDINGS_PAST_DEVICE);
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
        CubeGatedDeltaWalkRoutine
            .prepare(hw, self, &CubeGatedDeltaWalkRoutineSettings::FLOOR)
            .map(|_| ())
    }

    /// Every axis of the problem at its extent: what a routine partitions. The kept states sit
    /// outside the state's rows and columns, so each is a whole state.
    pub fn space(&self) -> Space {
        let heads = &self.heads;
        Space::new(&[
            (BATCH, self.batch),
            (POSITION, self.positions),
            (KEY_HEAD, heads.key_heads),
            (GROUP, heads.group),
            (KEPT_STATE, self.kept.get()),
            (KEY_FEATURE, heads.key_features),
            (VALUE_FEATURE, heads.value_features),
        ])
    }

    /// A problem of dense operands, each starting where its buffer does: what a key describes, and
    /// what a routine's pure tests prepare against.
    pub fn dense(
        batch: usize,
        positions: usize,
        heads: GatedDeltaHeads,
        output_element: ElemType,
        kept: NonZeroUsize,
    ) -> Self {
        let rows =
            |shape: &[usize]| ElemAlignment::new(shape, &row_major_contiguous_strides(shape));
        let GatedDeltaHeads {
            key_heads,
            group,
            key_features,
            value_features,
        } = heads;
        Self {
            batch,
            positions,
            heads,
            output_element,
            kept,
            alignments: GatedDeltaWalkOperandMap {
                query: rows(&[batch, positions, key_heads, key_features]),
                key: rows(&[batch, positions, key_heads, key_features]),
                value: rows(&[batch, positions, key_heads, group, value_features]),
                decay: rows(&[batch, positions, key_heads, group]),
                strength: rows(&[batch, positions, key_heads, group]),
                state: rows(&[batch, key_heads, group, key_features, value_features]),
                out: rows(&[batch, positions, key_heads, group, value_features]),
                kept: rows(&[
                    kept.get(),
                    batch,
                    key_heads,
                    group,
                    key_features,
                    value_features,
                ]),
            },
            start_bytes: GatedDeltaWalkOperandMap::splat(0),
        }
    }

    /// Bytes of one state.
    fn state_bytes(&self) -> usize {
        self.heads.key_features * self.heads.value_features * GATED_DELTA_WALK_ELEMENT.size()
    }
}

impl EstimatedProblem for GatedDeltaWalkProblem {
    /// Each key head's queries and keys, each head's values, scalars and outputs, the state read
    /// once and each kept state written once; per head and position, the decay, the recall, the
    /// write and the read are each a pass over the state.
    fn demand(&self) -> Work {
        let heads = &self.heads;
        let (value_heads, positions) = (self.batch * heads.value_heads(), self.positions);
        let input_bytes = GATED_DELTA_WALK_ELEMENT.size();
        let keyed = self.batch * heads.key_heads * positions * 2 * heads.key_features;
        let valued = value_heads * positions * (heads.value_features + 2);
        let read = value_heads * positions * heads.value_features * self.output_element.size();
        let states = value_heads * (1 + self.kept.get()) * self.state_bytes();
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
    use cubek_microkernel::gated_delta::walk::cube::CubeGatedDeltaWalkMicrokernel;
    use cubek_std::devices;
    use cubek_std::microkernel::Microkernel;

    fn sixteen_heads() -> GatedDeltaHeads {
        GatedDeltaHeads {
            key_heads: 16,
            group: 3,
            key_features: 128,
            value_features: 128,
        }
    }

    fn walk(heads: GatedDeltaHeads) -> GatedDeltaWalkProblem {
        GatedDeltaWalkProblem::dense(1, 8, heads, GATED_DELTA_WALK_ELEMENT, NonZeroUsize::MIN)
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
                Err(GATED_DELTA_WALK_BINDINGS_PAST_DEVICE),
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
        devices::topology(&mut hw).max_bindings = GATED_DELTA_WALK_BINDING_BUFFERS - 1;
        assert_eq!(
            problem.admit(&hw),
            Err(GATED_DELTA_WALK_BINDINGS_PAST_DEVICE)
        );
        devices::topology(&mut hw).max_bindings = GATED_DELTA_WALK_BINDING_BUFFERS;
        assert_eq!(problem.admit(&hw), Ok(()));
    }

    /// An output element that is packed or an integer is declined as not a float, a float the walk
    /// is not verified at as unverified, and the activations' floats are admitted; an operand
    /// other than `f32` is declined the same way.
    #[test]
    fn elements_that_are_not_plain_floats_are_declined() {
        use cubecl::ir::UIntKind;
        let (f16, f32) = (ElemType::Float(FloatKind::F16), GATED_DELTA_WALK_ELEMENT);
        let packed = ElemType::UInt(UIntKind::U32);
        let f64 = ElemType::Float(FloatKind::F64);
        for element in [f16, ElemType::Float(FloatKind::BF16), f32] {
            assert_eq!(GatedDeltaWalkProblem::admit_output(element), Ok(()));
        }
        assert_eq!(
            GatedDeltaWalkProblem::admit_output(packed),
            Err(GATED_DELTA_WALK_ELEMENT_NOT_FLOAT)
        );
        assert_eq!(
            GatedDeltaWalkProblem::admit_output(f64),
            Err(GATED_DELTA_WALK_ELEMENT_UNVERIFIED)
        );
        assert_eq!(GatedDeltaWalkProblem::admit_operand(f32), Ok(()));
        assert_eq!(
            GatedDeltaWalkProblem::admit_operand(f16),
            Err(GATED_DELTA_WALK_ELEMENT_UNVERIFIED)
        );
        assert_eq!(
            GatedDeltaWalkProblem::admit_operand(packed),
            Err(GATED_DELTA_WALK_ELEMENT_NOT_FLOAT)
        );
    }

    /// Heads with no feature or no head describe no walk.
    #[test]
    fn heads_that_describe_no_walk_are_declined() {
        let hw = devices::h100();
        let heads = sixteen_heads();
        for heads in [
            GatedDeltaHeads {
                key_heads: 0,
                ..heads
            },
            GatedDeltaHeads { group: 0, ..heads },
            GatedDeltaHeads {
                key_features: 0,
                ..heads
            },
            GatedDeltaHeads {
                value_features: 0,
                ..heads
            },
        ] {
            assert_eq!(
                walk(heads).admit(&hw),
                Err(GATED_DELTA_WALK_HEADS_EMPTY),
                "{heads:?}"
            );
        }
    }

    /// A head wider than a cube's units hold at the fewest registers is declined by the floor's
    /// refusal, not prepared into a launch the device refuses.
    #[test]
    fn a_head_too_wide_for_the_floor_is_declined() {
        let hw = super::fixtures::binding_enough(devices::wgpu_no_subgroups());
        let wide = GatedDeltaHeads {
            key_heads: 1,
            group: 1,
            key_features: 1 << 20,
            value_features: 4,
        };
        assert_eq!(
            walk(wide).admit(&hw),
            Err(Refusal::Declined {
                why: CubeGatedDeltaWalkMicrokernel::DECLINE
            })
        );
        assert_eq!(walk(sixteen_heads()).admit(&hw), Ok(()));
    }
}
