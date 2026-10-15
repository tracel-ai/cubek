//! The manual-mma leaf: `acc += lhs · rhs` via
//! [`MmaDefinition::execute`](cubecl::cmma::MmaDefinition).

use cubecl::cmma::{MatrixIdent, MatrixLayout};
use cubecl::prelude::*;

use crate::*;

#[cube]
impl MmaData {
    /// `self += lhs · rhs`; memory-window operands are loaded into transient fragments each call.
    pub(crate) fn mma<A: Numeric, L: Numeric, R: Numeric>(&mut self, lhs: &Tile, rhs: &Tile) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let io = comptime!(self.io);

        match &mut self.fragment {
            MmaFragment::Acc(acc) => match (&lhs.kind, &rhs.kind) {
                (TileKind::PlaneTile(a), TileKind::PlaneTile(b)) => match (a, b) {
                    (PlaneTile::Mma(a), PlaneTile::Mma(b)) => match (&a.fragment, &b.fragment) {
                        (MmaFragment::Lhs(af), MmaFragment::Rhs(bf)) => mma_execute::<L, R, A>(
                            af.at::<L>(),
                            bf.at::<R>(),
                            acc.at_mut::<A>(),
                            m,
                            n,
                            k,
                        ),
                        (
                            MmaFragment::LhsBlockScaled(af, a_scales),
                            MmaFragment::RhsBlockScaled(bf, b_scales),
                        ) => mma_execute_block_scaled::<A>(
                            af,
                            a_scales,
                            bf,
                            b_scales,
                            acc.at_mut::<A>(),
                            m,
                            n,
                            k,
                        ),
                        _ => panic!(
                            "MmaData::mma: operands must carry the Lhs/Rhs roles, both \
                             block-scaled or neither"
                        ),
                    },
                    _ => panic!("MmaData::mma: operands must be mma fragments"),
                },
                (TileKind::PlaneTile(a), TileKind::Memory(_)) => {
                    // An accumulator contracted again, read as the left factor in its registers:
                    // only the right one loads.
                    let (_, rhs_layout) =
                        comptime!(window_layouts(&lhs.place.space, &rhs.place.space));
                    let mut rb =
                        MmaData::operand::<R>(MatrixIdent::B, m, n, k, rhs_layout, io, false);
                    rb.load_window::<R>(rhs);
                    let b = PlaneTile::new_Mma(rb);
                    match (a, &b) {
                        (PlaneTile::Mma(a), PlaneTile::Mma(b)) => {
                            match (&a.fragment, &b.fragment) {
                                (MmaFragment::Acc(af), MmaFragment::Rhs(bf)) => {
                                    let af = accumulator_as_a::<L>(af.at::<L>(), m, n, k);
                                    mma_execute::<L, R, A>(
                                        &af,
                                        bf.at::<R>(),
                                        acc.at_mut::<A>(),
                                        m,
                                        n,
                                        k,
                                    )
                                }
                                _ => panic!(
                                    "MmaData::mma: a left factor held against a memory window is an \
                                 accumulator"
                                ),
                            }
                        }
                        _ => panic!("MmaData::mma: a held left factor is an mma fragment"),
                    }
                }
                (TileKind::Memory(_), TileKind::Memory(_)) => {
                    // Read each window in its stored order (`{n, k}` reads transposed).
                    let (lhs_layout, rhs_layout) =
                        comptime!(window_layouts(&lhs.place.space, &rhs.place.space));
                    let block_scaled = contracts_block_scaled::<L, R>(lhs, rhs, m, n, k);
                    let mut la = MmaData::operand::<L>(
                        MatrixIdent::A,
                        m,
                        n,
                        k,
                        lhs_layout,
                        io,
                        block_scaled,
                    );
                    la.load_window::<L>(lhs);
                    let mut rb = MmaData::operand::<R>(
                        MatrixIdent::B,
                        m,
                        n,
                        k,
                        rhs_layout,
                        io,
                        block_scaled,
                    );
                    rb.load_window::<R>(rhs);
                    match (&la.fragment, &rb.fragment) {
                        (MmaFragment::Lhs(af), MmaFragment::Rhs(bf)) => mma_execute::<L, R, A>(
                            af.at::<L>(),
                            bf.at::<R>(),
                            acc.at_mut::<A>(),
                            m,
                            n,
                            k,
                        ),
                        (
                            MmaFragment::LhsBlockScaled(af, a_scales),
                            MmaFragment::RhsBlockScaled(bf, b_scales),
                        ) => mma_execute_block_scaled::<A>(
                            af,
                            a_scales,
                            bf,
                            b_scales,
                            acc.at_mut::<A>(),
                            m,
                            n,
                            k,
                        ),
                        _ => panic!("MmaData::mma: transient operand fragments in the wrong roles"),
                    }
                }
                _ => panic!("MmaData::mma: operands must be fragments or memory windows"),
            },
            MmaFragment::Lhs(_)
            | MmaFragment::Rhs(_)
            | MmaFragment::LhsBlockScaled(..)
            | MmaFragment::RhsBlockScaled(..) => {
                panic!("MmaData::mma: the accumulator must carry the Acc role")
            }
        }
    }
}

/// Each factor's window layout: `A` row-major and `B` col-major when its trailing axis is
/// contracted, the opposite otherwise.
pub(crate) fn window_layouts(lhs: &Space, rhs: &Space) -> (MatrixLayout, MatrixLayout) {
    let trailing = |space: &Space| space.axis_at(space.rank() - 1);
    let lhs_layout = match rhs.contains(trailing(lhs)) {
        true => MatrixLayout::RowMajor,
        false => MatrixLayout::ColMajor,
    };
    let rhs_layout = match lhs.contains(trailing(rhs)) {
        true => MatrixLayout::ColMajor,
        false => MatrixLayout::RowMajor,
    };
    (lhs_layout, rhs_layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: Axis = Axis(0);
    const M: Axis = Axis(1);
    const N: Axis = Axis(2);
    const K: Axis = Axis(3);
    const TAPS: Axis = Axis(4);

    /// Layout follows whether each factor's trailing axis is contracted.
    #[test]
    fn a_window_reads_in_its_trailing_axis_order() {
        let lhs = Space::new(&[(B, 2), (M, 16), (K, 16)]);
        let row_weight = Space::new(&[(B, 2), (K, 16), (N, 16)]);
        let col_weight = Space::new(&[(B, 2), (N, 16), (K, 16)]);
        let lhs_transposed = Space::new(&[(B, 2), (K, 16), (M, 16)]);
        use MatrixLayout::{ColMajor, RowMajor};
        assert!(matches!(
            window_layouts(&lhs, &row_weight),
            (RowMajor, RowMajor)
        ));
        assert!(matches!(
            window_layouts(&lhs, &col_weight),
            (RowMajor, ColMajor)
        ));
        assert!(matches!(
            window_layouts(&lhs_transposed, &col_weight),
            (ColMajor, ColMajor)
        ));
    }

    /// Two contracted trailing axes read the same as one.
    #[test]
    fn a_window_contracted_over_two_axes_reads_as_over_one() {
        let lhs = Space::new(&[(M, 16), (TAPS, 3), (K, 16)]);
        let weight = Space::new(&[(N, 16), (TAPS, 3), (K, 16)]);
        use MatrixLayout::{ColMajor, RowMajor};
        assert!(matches!(
            window_layouts(&lhs, &weight),
            (RowMajor, ColMajor)
        ));
    }
}
