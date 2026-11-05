use crate::matmul::plane::PlaneFragmentRowsAlong;
use cubecl::tile::{Instruction, Semiring, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// How a product is contracted: the instruction its planes run, and the axis it lays its rows
/// along, which says the operand filling the instruction's `A`. Every sum or partial over the
/// product's operands is opened from it, so that order is decided in one place, whichever
/// plane, software or tensor-core, runs the instruction.
///
/// Its fields are public beside a method that opens a sum: it is comptime configuration a
/// kernel takes whole, as [`CubeStagedWalk`](crate::matmul::cube::CubeStagedWalk) is, and
/// guards no invariant its fields could break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulContraction {
    pub instruction: Instruction,
    pub rows_along: PlaneFragmentRowsAlong,
}

impl MatmulContraction {
    /// `instruction` with its rows along `m`, the lhs filling its `A`: what a plane whose rows are
    /// never laid otherwise, a software plane's, contracts through.
    pub const fn rows_along_m(instruction: Instruction) -> Self {
        Self {
            instruction,
            rows_along: PlaneFragmentRowsAlong::M,
        }
    }
}

#[cube]
impl MatmulContraction {
    /// An accumulator over `over`, in `E`, for the product of `a` by `b`: opened on the factor
    /// `contraction`'s rows lie along first, which fills the instruction's `A`. Not zeroed.
    pub fn accumulator<E: Numeric>(
        over: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] contraction: MatmulContraction,
    ) -> Tile {
        match contraction.rows_along {
            PlaneFragmentRowsAlong::M => {
                over.accumulator::<E>(a, b, contraction.instruction, Semiring::SUM_PROD)
            }
            PlaneFragmentRowsAlong::N => {
                over.accumulator::<E>(b, a, contraction.instruction, Semiring::SUM_PROD)
            }
        }
    }
}
