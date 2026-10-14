//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile, each descending to its leaf through
//! [`Descent::contract`].

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use crate::*;

#[cube]
impl Tile {
    /// `c = a · b`, under the semiring the accumulator was opened with, or a memory window stated
    /// [`accumulating`](Tile::accumulating). A memory window starts the sum fresh rather than
    /// reading the cell.
    pub fn mm(&mut self, lhs: &Tile, rhs: &Tile) {
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
    pub fn mma(&mut self, lhs: &Tile, rhs: &Tile) {
        let _served = naming::<Served>(self.element);
        let _lhs = naming::<Lhs>(lhs.element);
        let _rhs = naming::<Rhs>(rhs.element);
        Descent::contract::<Served, Lhs, Rhs>(self, lhs, rhs)
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
impl Tile {
    /// Commit the warpgroup MMAs this accumulator issued since its last commit, and return their
    /// completion: once it resolves, they are done reading what they read, the stages a walk
    /// holds for them among it. Only a warpgroup accumulator contracts asynchronously.
    pub fn commit(&self) -> Pending<()> {
        let _served = naming::<Served>(self.element);
        match &self.kind {
            TileKind::PlanePartition(p) => p.fragment().commit::<Served>(),
            TileKind::PlaneTile(t) => t.commit::<Served>(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!(
                "Tile::commit: only a plane group's warpgroup accumulator contracts asynchronously"
            ),
        }
    }
}

#[cube]
impl Tile {
    /// This memory window as a contraction's output: each unit's share tiled into `block`, folded
    /// under `semiring`. A plane-resident accumulator states both when it is opened.
    pub fn accumulating(
        &self,
        #[comptime] block: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) -> Tile {
        match &self.kind {
            TileKind::Memory(g) => {
                let mut g = g.clone();
                g.set_contraction(comptime!(Contraction { block, semiring }));
                Tile {
                    kind: TileKind::new_Memory(g),
                    factor: self.factor.clone(),
                    place: comptime!(self.place.clone()),
                    element: comptime!(self.element),
                }
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
impl PlaneTile {
    /// Contract this plane tile, each factor times whatever scales it carries.
    pub(crate) fn mma<E: Numeric, EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile,
        rhs: &Tile,
        #[comptime] out: Space,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                let transposed = transposed_rhs(lhs, rhs);
                strided_2d(lhs, rhs, comptime!(out.clone()), transposed);
                d.mma::<E, EL, ER>(lhs, rhs, out)
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
                d.mma::<E, EL, ER>(lhs, rhs)
            }
            PlaneTile::Wgmma(d) => {
                lhs.refuse_factor("PlaneTile::Wgmma");
                rhs.refuse_factor("PlaneTile::Wgmma");
                d.mma::<E, EL, ER>(lhs, rhs, out)
            }
            PlaneTile::Registers(d) => match &lhs.kind {
                TileKind::PlaneTile(block) => match block {
                    PlaneTile::Registers(block) => d.mma_block::<E, EL, ER>(block, rhs),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::PlanePartition(block) => match block.fragment() {
                    PlaneTile::Registers(block) => d.mma_block::<E, EL, ER>(&block, rhs),
                    PlaneTile::Cmma(_) | PlaneTile::Mma(_) | PlaneTile::Wgmma(_) => panic!(
                        "mma: a register block contracts a register block it holds, or memory"
                    ),
                },
                TileKind::Memory(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    let folded = comptime!(d.fold > 1);
                    strided_2d(lhs, rhs, comptime!(out.clone()), folded);
                    d.mma::<E, EL, ER>(lhs, rhs, out)
                }
            },
        }
    }
}

/// Asserts that operands are not gathered and read as one matrix each.
#[cube]
fn strided_2d(lhs: &Tile, rhs: &Tile, #[comptime] out: Space, #[comptime] rhs_along_k: bool) {
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
fn transposed_rhs(lhs: &Tile, rhs: &Tile) -> comptime_type!(bool) {
    match &rhs.kind {
        TileKind::PlaneTile(t) => match t {
            PlaneTile::Cmma(d) => comptime!(d.layout == MatrixLayout::ColMajor),
            PlaneTile::Mma(_) | PlaneTile::Registers(_) | PlaneTile::Wgmma(_) => comptime!(false),
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
fn flattened_k(lhs: &Tile, rhs: &Tile, #[comptime] out: Space) {
    comptime!(assert!(
        Space::contraction_agrees(&lhs.place.space, &rhs.place.space, &out),
        "mma: the operands list their contracted axes in different orders ({:?} against {:?}), \
         so their `k` edges do not line up",
        lhs.place.space.difference(&out),
        rhs.place.space.difference(&out)
    ));
}
