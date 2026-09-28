//! The manual-mma leaf, the raw-mma twin of [`cmma`](super::cmma): `acc += lhs · rhs` via
//! [`MmaDefinition::execute`](cubecl::cmma::MmaDefinition). The accumulator is a register
//! fragment; operands are fragments or smem/gmem windows, the latter loaded into `A`/`B` here.

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use crate::*;

#[cube]
impl<A: Numeric> MmaData<A> {
    /// Manual contraction `self += lhs · rhs`. Fragment operands execute directly; memory windows
    /// are loaded into transient `A`/`B` fragments each call, each in the order it lies.
    pub(crate) fn mma<L: Numeric, R: Numeric>(&mut self, lhs: &Tile<L>, rhs: &Tile<R>) {
        let m = comptime!(self.m);
        let n = comptime!(self.n);
        let k = comptime!(self.k);
        let io = comptime!(self.io);

        match &mut self.fragment {
            MmaFragment::Acc(acc) => match (&lhs.kind, &rhs.kind) {
                (TileKind::PlaneTile(a), TileKind::PlaneTile(b)) => match (a, b) {
                    (PlaneTile::Mma(a), PlaneTile::Mma(b)) => match (&a.fragment, &b.fragment) {
                        (MmaFragment::Lhs(af), MmaFragment::Rhs(bf)) => {
                            mma_execute::<L, R, A>(af, bf, acc, m, n, k)
                        }
                        _ => panic!("MmaData::mma: operands must carry the Lhs/Rhs roles"),
                    },
                    _ => panic!("MmaData::mma: operands must be mma fragments"),
                },
                (TileKind::Memory(_), TileKind::Memory(_)) => {
                    // Each window is read in the order it lies: the role's own, or its transpose
                    // where the contraction is not its trailing axis (a weight stored `{n, k}`).
                    let (lhs_layout, rhs_layout) =
                        comptime!(window_layouts(&lhs.place.space, &rhs.place.space));
                    let mut la = MmaData::<L>::lhs(m, n, k, lhs_layout, io);
                    la.load_window(lhs);
                    let mut rb = MmaData::<R>::rhs(m, n, k, rhs_layout, io);
                    rb.load_window(rhs);
                    match (&la.fragment, &rb.fragment) {
                        (MmaFragment::Lhs(af), MmaFragment::Rhs(bf)) => {
                            mma_execute::<L, R, A>(af, bf, acc, m, n, k)
                        }
                        _ => panic!("MmaData::mma: transient operand fragments in the wrong roles"),
                    }
                }
                _ => panic!("MmaData::mma: operands must be fragments or memory windows"),
            },
            MmaFragment::Lhs(_) | MmaFragment::Rhs(_) => {
                panic!("MmaData::mma: the accumulator must carry the Acc role")
            }
        }
    }
}

/// The layout each factor's window is read in: `A` row-major where its trailing axis is
/// contracted, `B` col-major there (the rule [`rhs_layout`](super::cmma::rhs_layout) states for
/// one contracted axis). A factor's trailing
/// axis is contracted exactly when the other factor holds it too, so a contraction over several
/// axes (a convolution's taps and channels, which the window's matrix edges flatten) reads the
/// same way as one over one, and a batch axis, never trailing, does not enter.
fn window_layouts(lhs: &Space, rhs: &Space) -> (MatrixLayout, MatrixLayout) {
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

    /// Each factor reads row- or col-major by whether its trailing axis is contracted: a col
    /// weight `{n, k}` col-major, a transposed lhs `{k, m}` col-major, a batch axis ignored.
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

    /// A contraction over taps and channels both, the weight stored with them trailing: both
    /// factors' two trailing axes are contracted, and each reads as over one.
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
