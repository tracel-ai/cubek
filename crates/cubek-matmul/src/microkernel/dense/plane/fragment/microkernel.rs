use crate::microkernel::dense::plane::{PlaneFragmentMicrokernel, PlaneFragmentStepSum};
use crate::microkernel::staging::StageRing;
use crate::microkernel::{MatmulOperands, MatmulRead};
use cubecl::tile::{Axis, Instruction, Region, Semiring, StageStorage, Stages, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// The partial a narrow step contracts into, opened over `out` beside the accumulator: one cell
/// of the operands' element for every cell the accumulator holds. None for a wide step.
#[cube]
pub(crate) fn plane_fragment_partial<EL: Numeric>(
    out: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] step_sum: PlaneFragmentStepSum,
    #[comptime] instruction: Instruction,
) -> ComptimeOption<Tile> {
    match comptime!(step_sum) {
        PlaneFragmentStepSum::Wide => ComptimeOption::<Tile>::new_None(),
        PlaneFragmentStepSum::Narrow => {
            ComptimeOption::new_Some(out.accumulator::<EL>(a, b, instruction, Semiring::SUM_PROD))
        }
    }
}

/// One plane's tile walked through stages of its own: the plane stages its operands a step of its
/// walk at a time, holding as many stages as `ring` says, and contracts every stage into `sum`
/// with the plane microkernel. `sum`, `c`, `a` and `b` are windowed to the plane; a narrow step's
/// partial is opened over `c`. A walk below the planes' level is the plane's own, so its stages
/// are too: each plane fills a copy of its own, met on its own barriers.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_staged_microkernel<EL: Numeric>(
    plane: &Region,
    sum: &Tile,
    c: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] microkernel: PlaneFragmentMicrokernel,
    #[comptime] stage: StageStorage,
    #[comptime] ring: StageRing,
) {
    let instruction = comptime!(microkernel.tile_instruction());
    let partial = plane_fragment_partial::<EL>(
        c,
        a,
        b,
        comptime!(microkernel.instruction.step_sum()),
        instruction,
    );
    let walk = plane.walk();
    let mut stages = Stages::smem(
        &walk,
        a,
        b,
        stage,
        comptime!(ring.stages_in_flight.get()),
        None,
    );
    stages.walk(walk, comptime!(ring.prefetch), |slot, step| {
        let sum_step = sum.at(step);
        slot.consume(|a_step, b_step| {
            plane_fragment_microkernel(
                step,
                &sum_step,
                &partial,
                a_step,
                b_step,
                instruction,
                None,
                comptime!(MatmulOperands::all(MatmulRead::AsBound)),
            );
        });
    });
}

/// One stage of one plane's tile: `a · b` added into `sum`, the operands staged in shared memory,
/// every instruction step of the stage and every fragment of the plane's grid.
///
/// `plane` is the plane's region of the stage; `sum`, `a` and `b` are already windowed to it, and
/// `partial` is [`plane_fragment_partial`](fn@plane_fragment_partial)'s, opened over the whole of the walker's output. The
/// loops are the same whichever instruction runs: below a step lie the plane's fragments, or the
/// one block of the software instruction this unit holds.
///
/// `held` is an axis the plane's steps walk that this product does not name, which each step
/// takes at its first coordinate: where a stage holds two products over one region, as
/// attention's score and value, each walks its own axes and holds the other's. Each leaf reads
/// its window of each operand as `reads` says.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn plane_fragment_microkernel(
    plane: &Region,
    sum: &Tile,
    partial: &ComptimeOption<Tile>,
    a: &Tile,
    b: &Tile,
    #[comptime] instruction: Instruction,
    #[comptime] held: Option<Axis>,
    #[comptime] reads: MatmulOperands<MatmulRead>,
) {
    #[comptime]
    match partial {
        ComptimeOption::None => wide_steps(plane, sum, a, b, instruction, held, reads),
        ComptimeOption::Some(partial) => narrow_steps(plane, sum, &partial.at(plane), a, b, reads),
    }
}

/// Every step contracted straight into `sum`: a grid of fragments in one call, which loads each
/// operand's fragments once; a register block, which is one unit's block and not a grid, cell by
/// cell. Each step takes `held` at its first coordinate.
#[cube]
fn wide_steps(
    plane: &Region,
    sum: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] instruction: Instruction,
    #[comptime] held: Option<Axis>,
    #[comptime] reads: MatmulOperands<MatmulRead>,
) {
    let steps = match comptime!(held) {
        Some(axis) => plane.walk().routed(axis, 0usize),
        None => plane.walk(),
    };
    for step in steps.unrolled() {
        let mut sum_step = sum.at(&step);
        let (a_step, b_step) = (a.at(&step), b.at(&step));
        match comptime!(instruction) {
            Instruction::Registers { .. } => {
                for cell in step.walk().unrolled() {
                    let mut sum_cell = sum_step.at(&cell);
                    let b_cell = MatmulRead::leaf(&sum_cell, &b_step.at(&cell), reads.rhs);
                    let a_cell = MatmulRead::leaf(&sum_cell, &a_step.at(&cell), reads.lhs);
                    sum_cell.mma(&a_cell, &b_cell);
                }
            }
            Instruction::Cmma | Instruction::Mma { .. } | Instruction::Wgmma => {
                let b_step = MatmulRead::leaf(&sum_step, &b_step, reads.rhs);
                let a_step = MatmulRead::leaf(&sum_step, &a_step, reads.lhs);
                sum_step.mma(&a_step, &b_step)
            }
        }
    }
}

/// Every step contracted into `partial`, emptied first, in the operands' element, and added into
/// `sum` at its end: the step's rounding never reaches the walk's. Only a register block promotes,
/// and a register block is one unit's block, so the step is walked cell by cell.
#[cube]
fn narrow_steps(
    plane: &Region,
    sum: &Tile,
    partial: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] reads: MatmulOperands<MatmulRead>,
) {
    for step in plane.walk().unrolled() {
        let (sum_step, partial_step) = (sum.at(&step), partial.at(&step));
        let (a_step, b_step) = (a.at(&step), b.at(&step));
        for cell in step.walk().unrolled() {
            let mut partial_cell = partial_step.at(&cell);
            let b_cell = MatmulRead::leaf(&partial_cell, &b_step.at(&cell), reads.rhs);
            let a_cell = MatmulRead::leaf(&partial_cell, &a_step.at(&cell), reads.lhs);
            partial_cell.mm(&a_cell, &b_cell);
            let mut sum_cell = sum_step.at(&cell);
            sum_cell.add_cast_from(&partial_cell);
        }
    }
}
