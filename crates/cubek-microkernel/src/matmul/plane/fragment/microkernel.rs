use crate::matmul::plane::{PlaneFragmentMicrokernel, PlaneFragmentStepSum};
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::{Axis, Instruction, Region, Semiring, Space, StageStorage, Stages, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::StageRing;

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
    match step_sum {
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
    let instruction = microkernel.tile_instruction();
    let step_sum = microkernel.instruction.step_sum();
    let partial = plane_fragment_partial::<EL>(c, a, b, step_sum, instruction);

    let walk = plane.walk();
    let in_flight = ring.stages_in_flight.get();
    let mut stages = Stages::smem(&walk, a, b, stage, in_flight, None);

    stages.walk(walk, ring.prefetch, |slot, step| {
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
                MatmulOperands::AS_BOUND,
            );
        });
    });
}

/// One stage of one plane's tile: `a · b` added into `sum`, the operands staged in shared memory,
/// every instruction step of the stage and every fragment of the plane's grid.
///
/// `plane` is the plane's region of the stage; `sum`, `a` and `b` are already windowed to it, and
/// `partial` is `plane_fragment_partial`'s, opened over the whole of the walker's output. The
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
        ComptimeOption::None => KWalk::wide_steps(plane, sum, a, b, instruction, held, reads),
        ComptimeOption::Some(partial) => {
            KWalk::narrow_steps(plane, sum, &partial.at(plane), a, b, reads)
        }
    }
}

/// How a plane walks its steps of `K`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum KWalk {
    /// Every step unrolled.
    Unrolled,
    /// The steps walk `K` alone: a runtime loop over them.
    Rolled,
    /// The steps also walk the sum's fragments: a runtime loop along this axis of `K`, each of its
    /// steps' fragments unrolled.
    Routed(Axis),
}

impl KWalk {
    /// The walk of a product whose steps take `held` at its first coordinate, out of operands
    /// laid over `a` and `b` into a sum over `sum`. `indexable` says both operands can be read at
    /// a runtime offset, which a fragment in registers cannot.
    fn new(held: Option<Axis>, indexable: bool, a: &Space, b: &Space, sum: &Space) -> Self {
        if !indexable {
            return KWalk::Unrolled;
        }

        let Some(held) = held else {
            return KWalk::Rolled;
        };

        let long = |space: &Space, axis| {
            space.contains(axis) && (space.is_dynamic(axis) || space.extent(axis) > 1)
        };

        let mut contracted = a
            .axes()
            .filter(|&axis| axis != held && long(a, axis) && long(b, axis) && !long(sum, axis));
        match (contracted.next(), contracted.next()) {
            (Some(k), None) => KWalk::Routed(k),
            _ => panic!(
                "KWalk: a product holding {held:?} contracts one axis of {a:?} and {b:?} into \
                 {sum:?}"
            ),
        }
    }
}

#[cube]
impl KWalk {
    /// Every step contracted straight into `sum`: a grid of fragments in one call, which loads each
    /// operand's fragments once; a register block, which is one unit's block and not a grid, cell by
    /// cell. Each step takes `held` at its first coordinate.
    fn wide_steps(
        plane: &Region,
        sum: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] instruction: Instruction,
        #[comptime] held: Option<Axis>,
        #[comptime] reads: MatmulOperands<MatmulRead>,
    ) {
        let (a_shared, b_shared) = (a.is_shared(), b.is_shared());
        let walk = comptime!(KWalk::new(
            held,
            a_shared && b_shared && instruction != Instruction::Wgmma,
            &a.space(),
            &b.space(),
            &sum.space()
        ));

        match walk {
            KWalk::Unrolled => {
                for step in KWalk::held_steps(plane, held).unrolled() {
                    KWalk::wide_step(&step, sum, a, b, instruction, reads);
                }
            }

            KWalk::Rolled => {
                for step in KWalk::held_steps(plane, held) {
                    KWalk::wide_step(&step, sum, a, b, instruction, reads);
                }
            }

            KWalk::Routed(k) => {
                let depth = KWalk::held_steps(plane, held).total()
                    / KWalk::held_steps(plane, held).routed(k, 0usize).total();
                for k_step in 0..depth {
                    for step in KWalk::held_steps(plane, held).routed(k, k_step).unrolled() {
                        KWalk::wide_step(&step, sum, a, b, instruction, reads);
                    }
                }
            }
        }
    }

    /// Every step contracted into `partial`, emptied first, in the operands' element, and added
    /// into `sum` at its end: the step's rounding never reaches the walk's. Only a register block
    /// promotes, and a register block is one unit's block, so the step is walked cell by cell.
    fn narrow_steps(
        plane: &Region,
        sum: &Tile,
        partial: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] reads: MatmulOperands<MatmulRead>,
    ) {
        let (a_shared, b_shared) = (a.is_shared(), b.is_shared());
        let walk = comptime!(KWalk::new(
            None,
            a_shared && b_shared,
            &a.space(),
            &b.space(),
            &sum.space()
        ));

        match walk {
            KWalk::Rolled => {
                for step in plane.walk() {
                    KWalk::narrow_step(&step, sum, partial, a, b, reads);
                }
            }

            KWalk::Unrolled | KWalk::Routed(_) => {
                for step in plane.walk().unrolled() {
                    KWalk::narrow_step(&step, sum, partial, a, b, reads);
                }
            }
        }
    }

    /// The steps `plane`'s walk takes, each at `held`'s first coordinate.
    fn held_steps(plane: &Region, #[comptime] held: Option<Axis>) -> Walk {
        match held {
            Some(axis) => plane.walk().routed(axis, 0usize),
            None => plane.walk(),
        }
    }

    /// One step of [`wide_steps`](Self::wide_steps).
    fn wide_step(
        step: &Region,
        sum: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] instruction: Instruction,
        #[comptime] reads: MatmulOperands<MatmulRead>,
    ) {
        let mut sum_step = sum.at(step);
        let (a_step, b_step) = (a.at(step), b.at(step));

        match instruction {
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

    /// One step of [`narrow_steps`](Self::narrow_steps).
    fn narrow_step(
        step: &Region,
        sum: &Tile,
        partial: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] reads: MatmulOperands<MatmulRead>,
    ) {
        let (sum_step, partial_step) = (sum.at(step), partial.at(step));
        let (a_step, b_step) = (a.at(step), b.at(step));

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
