use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::{Axis, Instruction, Region, Space, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// One stage of one plane's tile: `a · b` added into `sum`, the operands staged in shared memory,
/// every instruction step of the stage and every fragment of the plane's grid, through the
/// instruction the sum was opened in.
///
/// `plane` is the plane's region of the stage; `sum`, `a` and `b` are already windowed to it.
///
/// `held` is an axis the plane's steps walk that this product does not name, which each step
/// takes at its first coordinate: where a stage holds two products over one region, as
/// attention's score and value, each walks its own axes and holds the other's. Each leaf reads
/// its window of each operand as `reads` says.
#[cube]
pub fn plane_fragment_microkernel(
    plane: &Region,
    sum: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] held: Option<Axis>,
    #[comptime] reads: MatmulOperands<MatmulRead>,
) {
    KWalk::steps(plane, sum, a, b, held, reads);
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
    /// Every step contracted straight into `sum`, a grid of fragments in one call, which loads
    /// each operand's fragments once. Each step takes `held` at its first coordinate.
    fn steps(
        plane: &Region,
        sum: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] held: Option<Axis>,
        #[comptime] reads: MatmulOperands<MatmulRead>,
    ) {
        let (a_shared, b_shared) = (a.is_shared(), b.is_shared());
        let instruction = sum.sum_instruction();
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
                    KWalk::step(&step, sum, a, b, reads);
                }
            }
            KWalk::Rolled => {
                for step in KWalk::held_steps(plane, held) {
                    KWalk::step(&step, sum, a, b, reads);
                }
            }
            KWalk::Routed(k) => {
                let depth = KWalk::held_steps(plane, held).total()
                    / KWalk::held_steps(plane, held).routed(k, 0usize).total();
                for k_step in 0..depth {
                    for step in KWalk::held_steps(plane, held).routed(k, k_step).unrolled() {
                        KWalk::step(&step, sum, a, b, reads);
                    }
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

    /// One step of [`steps`](Self::steps).
    fn step(
        step: &Region,
        sum: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] reads: MatmulOperands<MatmulRead>,
    ) {
        let mut sum_step = sum.at(step);
        let b_step = MatmulRead::leaf(&sum_step, &b.at(step), reads.rhs);
        let a_step = MatmulRead::leaf(&sum_step, &a.at(step), reads.lhs);
        sum_step.mma(&a_step, &b_step)
    }
}
