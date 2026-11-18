//! The problem a routine computes.

use crate::kernels::RMS_NORM_ELEMENT_UNVERIFIED;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Space;
use cubecl::tile::launch::Refusal;
use cubecl::tune::Work;
use cubek_matmul::kernels::tensor::{ELEMENT_NOT_ON_DEVICE, Placement};
use cubek_microkernel::rms_norm::{B, GateOrder, RmsNormOperand, RmsNormOperands, RmsNormWorkItem};
use cubek_microkernel::row_store::RowStore;
use cubek_std::Hardware;
use cubek_std::microkernel::{CubeStaging, GlobalStream};
use cubek_std::routine::EstimatedProblem;

/// What a launch declines with where the output is written at an element other than the input's
/// or `f32`, or at `f32` into anything but a buffer: a packed, tiled or sink output holds the
/// input's element.
pub const RMS_NORM_OUTPUT_UNSUPPORTED: Refusal = Refusal::Declined {
    why: "the RMS norm writes f32 or the input's element into a buffer it allocates, and the input's element into any other output",
};

const F32: ElemType = ElemType::Float(FloatKind::F32);

/// A gate's order and where its view lies: what a gated problem states of its gate operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormGateOperand {
    pub order: GateOrder,
    pub placement: Placement,
}

/// `out[b, r] = input[b, r] / sqrt(mean(input[b, r]²) + epsilon) · weight` over `batch` entries
/// of `rows` rows of `dim` elements, as the host describes it, the rows gated by `silu(gate)` where
/// [`gate`](Self::gate) says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmsNormRoutineProblem {
    pub batch: usize,
    pub rows: usize,
    pub dim: usize,
    /// The element of the input and the weight.
    pub element: ElemType,
    /// The element the output is written at: `element`, or `f32`.
    pub output_element: ElemType,
    /// Where each operand every norm binds lies.
    pub placements: RmsNormOperands<Placement>,
    /// How the output's rows are stored.
    pub store: RowStore,
    /// The gate operand, where the rows are gated: where it enters and where it lies.
    pub gate: Option<RmsNormGateOperand>,
}

impl RmsNormRoutineProblem {
    /// Whether no row is computed: a grouped call over no tokens has rows in its entries but no
    /// entry.
    pub fn is_empty(&self) -> bool {
        self.batch * self.rows == 0
    }

    /// The order the gate enters at, where the rows are gated.
    pub fn gate_order(&self) -> Option<GateOrder> {
        self.gate.map(|gate| gate.order)
    }

    /// Each operand the norm binds with where it lies, in the order [`RmsNormOperand::of`] names
    /// them: the gate only where the rows are gated.
    pub fn bound_placements(&self) -> Vec<(RmsNormOperand, Placement)> {
        [
            (RmsNormOperand::Input, self.placements.input),
            (RmsNormOperand::Weight, self.placements.weight),
        ]
        .into_iter()
        .chain(self.gate.map(|gate| (RmsNormOperand::Gate, gate.placement)))
        .chain([(RmsNormOperand::Out, self.placements.out)])
        .collect()
    }

    /// What every microkernel of this problem is handed.
    pub fn work_item(&self) -> RmsNormWorkItem {
        RmsNormWorkItem {
            element: self.element,
            output_element: self.output_element,
            dim: self.dim,
            store: self.store,
            gate: self.gate_order(),
        }
    }

    /// Every axis of the problem at its extent, its rows and its dim as the store states them:
    /// what a routine partitions.
    pub fn space(&self) -> Space {
        let axes = [
            vec![(B, self.batch)],
            self.store.rows_axes(self.rows),
            self.store.width_axes(self.dim),
        ]
        .concat();
        Space::new(&axes)
    }

    /// How one cube walks `rows` ([`RmsNormCubeRows`]): every row read once and written once, the
    /// weight read once, in one stage held while it computes.
    pub fn cube_staging(&self, rows: RmsNormCubeRows) -> CubeStaging {
        let gated = usize::from(self.gate.is_some());
        let stream = |elements: usize, element: ElemType| GlobalStream {
            bytes: (elements * element.size()) as u64,
            run_bytes: rows.run * element.size(),
        };
        let mut reads = vec![
            stream(rows.rows * self.dim, self.element),
            stream(self.dim, self.element),
        ];
        if self.gate.is_some() {
            reads.push(stream(rows.rows * self.dim, self.element));
        }
        CubeStaging {
            operations: ((4 + 4 * gated) * rows.rows * self.dim) as f64,
            reads,
            writes: vec![stream(rows.rows * self.dim, self.output_element)],
            stages: 1,
            stages_in_flight: 1,
            drain_atomics: 0,
        }
    }

    /// A problem of dense operands of `element`, their views starting where their buffers do, one
    /// batch entry holding every row: what a key describes, and what a routine's pure tests
    /// prepare against.
    pub fn dense(rows: usize, dim: usize, element: ElemType) -> Self {
        let row = Placement::dense_rows(dim);
        Self {
            batch: 1,
            rows,
            dim,
            element,
            output_element: element,
            placements: RmsNormOperands {
                input: row,
                weight: row,
                out: row,
            },
            store: RowStore::Plain,
            gate: None,
        }
    }

    /// This problem, its rows stored as `store` says.
    pub fn stored(self, store: RowStore) -> Self {
        Self { store, ..self }
    }

    /// Whether `hw` runs a norm of rows of `element` written at `output_element`.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_ELEMENT_UNVERIFIED`] for rows that are not floats,
    /// [`RMS_NORM_OUTPUT_UNSUPPORTED`] for an output other than `f32` and the rows' element,
    /// [`ELEMENT_NOT_ON_DEVICE`] for an element the device does not run.
    pub fn admit_elements(
        hw: &Hardware,
        element: ElemType,
        output_element: ElemType,
    ) -> Result<(), Refusal> {
        if !matches!(element, ElemType::Float(_)) {
            return Err(RMS_NORM_ELEMENT_UNVERIFIED);
        }
        if output_element != element && output_element != F32 {
            return Err(RMS_NORM_OUTPUT_UNSUPPORTED);
        }
        match hw.runs(element) && hw.runs(output_element) {
            true => Ok(()),
            false => Err(ELEMENT_NOT_ON_DEVICE),
        }
    }

    /// Whether `hw` runs this problem's elements and its output's store: an output written at
    /// `f32` goes into a plain buffer only.
    ///
    /// # Errors
    ///
    /// As [`admit_elements`](Self::admit_elements), and [`RMS_NORM_OUTPUT_UNSUPPORTED`] for an
    /// `f32` output into a packed store.
    pub fn admit(&self, hw: &Hardware) -> Result<(), Refusal> {
        Self::admit_elements(hw, self.element, self.output_element)?;
        match self.store == RowStore::Plain || self.output_element == self.element {
            true => Ok(()),
            false => Err(RMS_NORM_OUTPUT_UNSUPPORTED),
        }
    }
}

/// The rows one cube owns whole, and the run of a row each of its planes touches contiguously.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RmsNormCubeRows {
    pub rows: usize,
    /// Consecutive elements of a row one owner reads: the row, or a cube's plane's run of it.
    pub run: usize,
}

impl EstimatedProblem for RmsNormRoutineProblem {
    /// Each row read once, and its gate where there is one, written once, the weight read once, and
    /// four operations an element, and four more for the silu and its product where gated.
    fn demand(&self) -> Work {
        let gated = usize::from(self.gate.is_some());
        let elements = self.batch * self.rows * self.dim;
        Work {
            compute_ops: (4 + 4 * gated) * elements,
            bytes: ((1 + gated) * elements + self.dim) * self.element.size()
                + elements * self.output_element.size(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use cubecl::features::TypeUsage;
    use cubecl::ir::UIntKind;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// A device running `f16` and `f32`.
    pub(crate) fn running(mut hw: Hardware) -> Hardware {
        for element in [F16, F32] {
            *devices::features(&mut hw)
                .types
                .elem
                .entry(element)
                .or_default() |= TypeUsage::Arithmetic | TypeUsage::Buffer;
        }
        hw
    }

    /// Float rows are written at their own element or `f32`; any other output, rows that are not
    /// floats and an element the device does not run are declined.
    #[test]
    fn the_norm_is_written_at_the_input_element_or_f32() {
        let hw = running(devices::h100());
        let admit = |element, output| RmsNormRoutineProblem::admit_elements(&hw, element, output);
        let (bf16, u32) = (
            ElemType::Float(FloatKind::BF16),
            ElemType::UInt(UIntKind::U32),
        );
        assert_eq!(admit(F16, F16), Ok(()));
        assert_eq!(admit(F16, F32), Ok(()));
        assert_eq!(admit(F16, bf16), Err(RMS_NORM_OUTPUT_UNSUPPORTED));
        assert_eq!(admit(u32, u32), Err(RMS_NORM_ELEMENT_UNVERIFIED));
        assert_eq!(admit(bf16, bf16), Err(ELEMENT_NOT_ON_DEVICE));
    }
}
