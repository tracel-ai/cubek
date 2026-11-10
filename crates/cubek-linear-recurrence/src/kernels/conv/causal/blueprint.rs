//! A causal conv launch's compile-time decisions.

use crate::kernels::conv::CausalConvProblem;
use cubecl::client::Client;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubecl::tile::{Launcher, Levels};
use cubek_microkernel::linear_recurrence::conv::CausalConvOperand;
use cubek_microkernel::linear_recurrence::conv::cube::{
    CausalConvSettings, CubeCausalConvMicrokernel,
};
use cubek_microkernel::linear_recurrence::{CHANNEL_BOX, CausalConvPartitioning, LINE_IN_BOX};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{CubeResources, Grid};

/// What a causal conv launch compiles: the microkernel, the partitioning, and the width the
/// channels bind at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalConvBlueprint {
    pub microkernel: CubeCausalConvMicrokernel,
    pub partitioning: CausalConvPartitioning,
    pub width: usize,
}

impl CausalConvBlueprint {
    /// The launch of `problem` on `hw`: no cube holds more units than the channels have lines.
    pub(crate) fn prepare(
        hw: &Hardware,
        problem: &CausalConvProblem,
        settings: &CausalConvSettings,
    ) -> Result<(Self, Grid), Refusal> {
        problem.admit_launch(hw)?;
        let item = problem.work_item();
        let provisional = CubeCausalConvMicrokernel::best(hw, &item, settings)?;
        let width = Self::settled_width(hw, problem, &provisional);
        let lines = problem.channels / width;
        let microkernel = CubeCausalConvMicrokernel {
            units: provisional.units.min(lines),
            ..provisional
        };
        let partitioning = CausalConvPartitioning::new(
            problem.space(microkernel.units, width),
            Levels::leaf(&problem.leaf(width)).units_distributed(LINE_IN_BOX, microkernel.units),
            CHANNEL_BOX,
        );
        let cube = CubeResources {
            plane_width: microkernel.units as u32,
            shared_memory: 0,
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
        problem: &CausalConvProblem,
        microkernel: &CubeCausalConvMicrokernel,
    ) -> usize {
        let element = |operand| match operand {
            CausalConvOperand::Output => problem.output_element,
            _ => problem.element,
        };
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        for operand in CausalConvOperand::CHANNELS {
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
        vectorization.width(CausalConvOperand::Input, |operand| {
            hw.vector_ceiling(element(operand))
        })
    }
}
