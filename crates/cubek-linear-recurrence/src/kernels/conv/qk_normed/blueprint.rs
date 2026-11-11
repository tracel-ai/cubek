//! A conv launch's compile-time decisions.

use crate::kernels::conv::{
    ConvExtent, QK_NORMED_CONV_OUTPUT_ELEMENT, QkNormedConvProblem, RingWriteBack,
};
use cubecl::tile::Levels;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::linear_recurrence::conv::QkNormedConvOperand;
use cubek_microkernel::linear_recurrence::conv::cube::{
    CubeQkNormedConvMicrokernel, QkNormedConvSettings,
};
use cubek_microkernel::linear_recurrence::{CausalConvPartitioning, HEAD_CHANNEL, KEY_HEAD};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{CubeResources, Grid};
use std::sync::Arc;

/// What a conv launch compiles: the microkernel, the partitioning, the width the channels bind at,
/// and the launch that fills a prompt's rings after it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QkNormedConvBlueprint {
    pub microkernel: CubeQkNormedConvMicrokernel,
    pub partitioning: CausalConvPartitioning,
    pub width: usize,
    pub write_back: Option<Arc<RingWriteBack>>,
}

impl QkNormedConvBlueprint {
    /// The launch of `problem` on `hw`: no cube holds more units than a head has lines, and the write
    /// back of a prompt is checked against the device too.
    pub(crate) fn prepare(
        hw: &Hardware,
        problem: &QkNormedConvProblem,
        settings: &QkNormedConvSettings,
    ) -> Result<(Self, Grid), Refusal> {
        problem.admit_launch(hw)?;
        let item = problem.work_item();
        let provisional = CubeQkNormedConvMicrokernel::best(hw, &item, settings)?;
        let width = Self::settled_width(hw, problem, &provisional);
        let lines = item.heads.channels_per_key_head() / width;
        let microkernel = CubeQkNormedConvMicrokernel {
            units: provisional.units.min(lines),
            ..provisional
        };
        let partitioning = CausalConvPartitioning::new(
            problem.space(),
            Levels::leaf(&problem.leaf(width)).units_distributed(HEAD_CHANNEL, lines),
            &problem.past.cube_axes(KEY_HEAD),
        );
        let cube = CubeResources {
            plane_width: microkernel.units as u32,
            shared_memory: microkernel.shared_memory(),
        };
        let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
        let extent = ConvExtent {
            batch: problem.batch,
            channels: item.heads.conv_dim(),
            taps: problem.taps,
        };
        let write_back = RingWriteBack::prepare(hw, &problem.past, extent, width)?;
        let blueprint = Self {
            microkernel,
            partitioning,
            width,
            write_back,
        };
        Ok((blueprint, grid))
    }

    /// The widest width the device, the microkernel's accesses and every operand's alignment and
    /// view start admit.
    fn settled_width(
        hw: &Hardware,
        problem: &QkNormedConvProblem,
        microkernel: &CubeQkNormedConvMicrokernel,
    ) -> usize {
        let element = |operand| match operand {
            QkNormedConvOperand::Query | QkNormedConvOperand::Key | QkNormedConvOperand::Value => {
                QK_NORMED_CONV_OUTPUT_ELEMENT
            }
            _ => problem.element,
        };
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        for operand in QkNormedConvOperand::CHANNELS {
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: problem.alignments.get(operand),
            });
            vectorization.require(VectorRequirement::Aligned {
                operand,
                alignment: ElemAlignment::at_start(
                    problem.start_bytes.get(operand),
                    element(operand).size(),
                ),
            });
        }
        vectorization.width(QkNormedConvOperand::Qkv, |operand| {
            hw.vector_ceiling(element(operand))
        })
    }
}
