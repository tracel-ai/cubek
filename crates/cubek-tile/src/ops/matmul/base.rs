//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile, each descending to its leaf through
//! [`Descent::contract`].

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b`, under the semiring the accumulator was opened with, or a memory window stated
    /// [`accumulating`](Tile::accumulating). A memory window starts the sum fresh rather than
    /// reading the cell.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        let in_memory = self.is_memory();
        if comptime!(in_memory) {
            // A clone holds the same window: reading it reads `self`.
            let acc = self.clone();
            let contraction = match &acc.kind {
                TileKind::Memory(g) => g.stated_contraction(),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("Tile::mm: only a memory window states its contraction")
                }
            };
            let semiring = contraction.semiring;
            let init_from = self.request_init_from(comptime!(InitFrom::Identity));
            match comptime!(init_from) {
                InitFrom::Identity => {}
                InitFrom::Cell => self.init_identity(comptime!(semiring.add())),
            }
            self.mma(lhs, rhs);
            self.request_init_from(comptime!(InitFrom::Cell));
        } else {
            self.reset();
            self.mma(lhs, rhs);
        }
    }

    /// `c += a · b`, folding onto what the accumulator or memory window holds.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        Descent::contract(self, lhs, rhs)
    }

    /// This plane-resident accumulator back at its semiring's identity, as it was opened.
    pub fn reset(&mut self) {
        // A clone holds the same fragments: reading it reads `self`.
        let acc = self.clone();
        let semiring = match &acc.kind {
            TileKind::PlaneTile(t) => t.semiring(),
            TileKind::PlanePartition(p) => p.at(0usize, 0usize).semiring(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::reset: only a plane-resident accumulator has an identity to return to"
                )
            }
        };
        self.init_identity(comptime!(semiring.add()));
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// This memory window as a contraction's output: each unit's share tiled into `block`, folded
    /// under `semiring`. A plane-resident accumulator states both when it is opened.
    pub fn accumulating(
        &self,
        #[comptime] block: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) -> Tile<Acc> {
        match &self.kind {
            TileKind::Memory(g) => {
                let mut g = g.clone();
                g.set_contraction(comptime!(Contraction { block, semiring }));
                Tile::new(TileKind::new_Memory(g), comptime!(self.place.clone()))
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!(
                "Tile::accumulating: a memory window states how it is contracted into; a \
                 plane-resident accumulator states it when opened (Tile::accumulator)"
            ),
        }
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// Contract this plane tile, each factor times whatever scales it carries.
    pub(crate) fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                let transposed = transposed_rhs(lhs, rhs);
                strided_2d(lhs, rhs, comptime!(out.clone()), transposed);
                d.mma(lhs, rhs, out)
            }
            PlaneTile::Mma(d) => {
                // The manual-mma instruction reads registers, where a scaled factor cannot land,
                // unless the device's block-scaled instruction takes the scales beside the values.
                let (m, n, k) = comptime!((d.m, d.n, d.k));
                let block_scaled = contracts_block_scaled(lhs, rhs, m, n, k);
                if comptime!(!block_scaled) {
                    lhs.refuse_factor("PlaneTile::Mma");
                    rhs.refuse_factor("PlaneTile::Mma");
                }
                flattened_k(lhs, rhs, out);
                d.mma(lhs, rhs)
            }
            PlaneTile::Registers(d) => match &lhs.kind {
                TileKind::PlaneTile(block) => match block {
                    PlaneTile::Registers(block) => d.mma_block(block, rhs),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::PlanePartition(block) => match block.fragment() {
                    PlaneTile::Registers(block) => d.mma_block(&block, rhs),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::Memory(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    let folded = comptime!(d.fold > 1);
                    strided_2d(lhs, rhs, comptime!(out.clone()), folded);
                    d.mma(lhs, rhs, out)
                }
            },
        }
    }
}

/// Asserts that operands are not gathered and read as one matrix each.
#[cube]
fn strided_2d<EL: Numeric, ER: Numeric>(
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    #[comptime] out: Space,
    #[comptime] rhs_along_k: bool,
) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let flat = comptime!({
        let kc = Space::merge(&[&lhs.place.space, &rhs.place.space]).contracted_extent(&out);
        let axes = MatrixAxes::accumulator(&out, &lhs.place.space);
        let cols = axes.cols(&out);
        let rhs_matrix = if rhs_along_k {
            MatrixAxes::new(&rhs.place.space, cols, kc)
        } else {
            MatrixAxes::new(&rhs.place.space, kc, cols)
        };
        MatrixAxes::new(&lhs.place.space, axes.rows(&out), kc).is_ok() && rhs_matrix.is_ok()
    });
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered && flat,
        "mma: a cmma or plane-register fragment reads one `k` edge off a directly addressed \
         operand; a gather, or a contraction these axes give no edge for, needs the manual-mma \
         leaf, or an unpromoted Gmem/Smem accumulator, whose software instruction is the \
         `memory` arm of `Descent::contract`"
    ));
}

/// Whether `rhs` is read col-major: a col-major cmma fragment or a staged `(col, k)` window.
#[cube]
fn transposed_rhs<EL: Numeric, ER: Numeric>(
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
) -> comptime_type!(bool) {
    match &rhs.kind {
        TileKind::PlaneTile(t) => match t {
            PlaneTile::Cmma(d) => comptime!(d.layout == MatrixLayout::ColMajor),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) => comptime!(false),
        },
        // The contracted axis is the lhs's trailing one; not every axis the output lacks is.
        // Only a shared stage can answer column-major; a gmem rhs never does.
        TileKind::Memory(m) => comptime!(
            m.address == AddressSpace::Shared
                && crate::ops::matmul::leaf::rhs_layout(
                    &rhs.place.space,
                    lhs.place.space.axis_at(lhs.place.space.rank() - 1)
                ) == MatrixLayout::ColMajor
        ),
        TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => comptime!(false),
    }
}

/// Asserts that operands contract their shared axes in the same order.
#[cube]
fn flattened_k<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    comptime!(assert!(
        Space::contraction_agrees(&lhs.place.space, &rhs.place.space, &out),
        "mma: the operands list their contracted axes in different orders ({:?} against {:?}), \
         so their `k` edges do not line up",
        lhs.place.space.difference(&out),
        rhs.place.space.difference(&out)
    ));
}
