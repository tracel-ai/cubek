//! Where a contraction runs, read off the tile it is called on ([`Descent`]).

use cubecl::prelude::*;

use crate::*;

/// Where `acc += lhs · rhs` runs, from the accumulator it is called on and the level below it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Descent {
    /// `acc` holds one fragment or block, or is a memory window: its instruction runs here.
    Here,
    /// `acc` is a grid of fragments and the level below steps the contraction: each of its regions
    /// contracts in turn.
    Steps,
    /// `acc` is a grid of fragments and the level below only moves its cells: each operand's
    /// fragments load once in the instruction's form, then every cell contracts.
    Cells(Instruction),
}

impl Descent {
    /// The descent into an accumulator placed at `acc`, a grid contracting through `grid`'s
    /// instruction or `None` where it holds one fragment, from operands spanning `lhs` and `rhs`.
    pub(crate) fn of(grid: Option<Instruction>, acc: &Placement, lhs: &Space, rhs: &Space) -> Self {
        let Some(instruction) = grid else {
            return Descent::Here;
        };
        let below = acc.below().first().unwrap_or_else(|| {
            panic!("mma: a grid of fragments contracts at a level with its cells below it")
        });
        let operands = Space::merge(&[lhs, rhs]);
        let steps_contraction = below
            .axes()
            .iter()
            .any(|&axis| operands.contains(axis) && !acc.space.contains(axis));
        match steps_contraction {
            true => Descent::Steps,
            false => Descent::Cells(instruction),
        }
    }
}

#[cube]
impl Descent {
    /// The descent a contraction into `acc` takes.
    pub(crate) fn of_tile<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
        acc: &Tile<E>,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
    ) -> comptime_type!(Descent) {
        match &acc.kind {
            TileKind::PlanePartition(p) => {
                let grid = p.grid();
                comptime!(Descent::of(
                    grid,
                    &acc.place,
                    &lhs.place.space,
                    &rhs.place.space
                ))
            }
            TileKind::PlaneTile(_)
            | TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(Descent::Here),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A 2×2 grid of 8×8 fragments, with `below` the level under it.
    fn grid_over(below: Level) -> Placement {
        Placement::new(Space::new(&[(M, 16), (N, 16)]), 0, vec![below])
    }

    fn descent(grid: Option<Instruction>, below: Level) -> Descent {
        let lhs = Space::new(&[(M, 16), (K, 16)]);
        let rhs = Space::new(&[(K, 16), (N, 16)]);
        Descent::of(grid, &grid_over(below), &lhs, &rhs)
    }

    /// One fragment runs its instruction where it is called, whatever lies below.
    #[test]
    fn one_fragment_contracts_here() {
        let below = Levels::leaf(&[(M, 8), (N, 8)])
            .walk(&[(M, 2), (N, 2)])
            .level();
        assert_eq!(descent(None, below), Descent::Here);
    }

    /// A level that steps only the grid's cells loads each operand's fragments once above it.
    #[test]
    fn a_grid_over_its_cells_loads_its_operands_once() {
        let below = Levels::leaf(&[(M, 8), (N, 8)])
            .walk(&[(M, 2), (N, 2)])
            .level();
        assert_eq!(
            descent(Some(Instruction::Cmma), below),
            Descent::Cells(Instruction::Cmma)
        );
    }

    /// A level that also steps `K` contracts each of its regions in turn, `K` included.
    #[test]
    fn a_grid_over_the_contraction_steps_it() {
        let below = Levels::leaf(&[(M, 8), (N, 8), (K, 8)])
            .walk(&[(M, 2), (N, 2), (K, 2)])
            .level();
        assert_eq!(descent(Some(Instruction::Cmma), below), Descent::Steps);
    }
}
