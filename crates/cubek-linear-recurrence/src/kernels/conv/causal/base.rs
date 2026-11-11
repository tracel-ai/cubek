//! How the causal conv places a launch and settles the width its channels bind at.

use crate::kernels::conv::{CausalConvBlueprint, CausalConvProblem, RingWriteBackLaunch};
use cubecl::tile::Levels;
use cubecl::tile::launch::{Refusal, VectorRequirement, Vectorization};
use cubecl::tile::layout::ElemAlignment;
use cubek_microkernel::linear_recurrence::conv::CausalConvOperand;
use cubek_microkernel::linear_recurrence::conv::cube::{
    CausalConvSettings, CubeCausalConvMicrokernel,
};
use cubek_microkernel::linear_recurrence::{CHANNEL_BOX, CausalConvPartitioning, LINE_IN_BOX};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_std::routine::{CubeResources, Grid};

/// The launch of `problem` on `hw`: no cube holds more units than the channels have lines, and the
/// write back of a prompt is checked against the device too.
pub(super) fn prepare(
    hw: &Hardware,
    problem: &CausalConvProblem,
    settings: &CausalConvSettings,
) -> Result<(CausalConvBlueprint<CubeCausalConvMicrokernel>, Grid), Refusal> {
    problem.admit_launch(hw)?;
    let item = problem.work_item();
    let provisional = CubeCausalConvMicrokernel::best(hw, &item, settings)?;
    let width = settled_width(hw, problem, &provisional);
    let lines = problem.channels / width;
    let microkernel = CubeCausalConvMicrokernel {
        units: provisional.units.min(lines),
        ..provisional
    };
    let partitioning = CausalConvPartitioning::new(
        problem.space(microkernel.units, width),
        Levels::leaf(&problem.leaf(width)).units_distributed(LINE_IN_BOX, microkernel.units),
        &problem.past.cube_axes(CHANNEL_BOX),
    );
    let cube = CubeResources {
        plane_width: microkernel.units as u32,
        shared_memory: 0,
    };
    let grid = Grid::new(hw, partitioning.as_cubek(), cube)?;
    let write_back = RingWriteBackLaunch::prepare(hw, &problem.past, problem.extent(), width)?;
    let blueprint = CausalConvBlueprint::new(microkernel, partitioning, width, write_back);
    Ok((blueprint, grid))
}

/// The widest width the device, the microkernel's accesses and every operand's alignment and view
/// start admit.
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
