//! A conv launch's compile-time decisions.

use crate::kernels::conv::{QK_NORMED_CONV_OUTPUT_ELEMENT, QkNormedConvProblem};
use cubecl::client::Client;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Levels};
use cubek_microkernel::linear_recurrence::conv::QkNormedConvOperand;
use cubek_microkernel::linear_recurrence::conv::cube::{
    CubeQkNormedConvMicrokernel, QkNormedConvSettings,
};
use cubek_microkernel::linear_recurrence::{CausalConvPartitioning, HEAD_CHANNEL, KEY_HEAD};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{CubeResources, Grid};

/// What a conv launch compiles: the microkernel, the partitioning, and the width the channels bind
/// at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QkNormedConvBlueprint {
    pub microkernel: CubeQkNormedConvMicrokernel,
    pub partitioning: CausalConvPartitioning,
    pub width: usize,
}

impl QkNormedConvBlueprint {
    /// The launch of `problem` on `hw`: no cube holds more units than a head has lines.
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
            KEY_HEAD,
        );
        let cube = CubeResources {
            plane_width: microkernel.units as u32,
            shared_memory: microkernel.shared_memory(),
        };
        let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
        let blueprint = Self {
            microkernel,
            partitioning,
            width,
        };
        Ok((blueprint, grid))
    }

    pub(crate) fn launcher(&self, client: &Client, grid: &Grid) -> Launcher {
        grid.launcher(client, || {
            (
                self.partitioning.kernel(),
                self.partitioning.as_cubek().space().clone(),
            )
        })
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
        for &operand in problem.past.channels() {
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
