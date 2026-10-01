//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile: the leaf dispatch ([`mma_leaf`]).

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use super::leaf::memory;
use crate::tile::base::witnessed_space;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b`, under the semiring the accumulator was opened with, or a memory window stated
    /// [`accumulating`](Tile::accumulating). A memory window starts the sum fresh rather than
    /// reading the cell.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        let in_memory = self.is_memory();
        if comptime!(in_memory) {
            let semiring = contraction_of(&self.clone()).semiring;
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
        mma_leaf(self, lhs, rhs)
    }

    /// This plane-resident accumulator back at its semiring's identity, as it was opened.
    pub fn reset(&mut self) {
        // A clone holds the same fragments: reading it reads `self`.
        let semiring = held_semiring(&self.clone());
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

/// The leaf contraction `acc += lhs · rhs`, dispatched on the accumulator's form.
#[cube]
pub(crate) fn mma_leaf<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
) {
    // A clone holds the same fragments: writes through it land in `acc`.
    let grid = acc.clone();
    let descent = Descent::of_tile::<E, Lhs, Rhs>(&grid, lhs, rhs);
    match comptime!(descent) {
        Descent::Here => mma_here::<E, Lhs, Rhs>(acc, lhs, rhs),
        Descent::Steps => mma_steps::<E, Lhs, Rhs>(&grid, lhs, rhs),
        Descent::Cells(instruction) => mma_cells::<E, Lhs, Rhs>(&grid, lhs, rhs, instruction),
    }
}

/// [`Descent::Steps`]: each region of the level below, walked over the whole contraction's box
/// (`acc`'s alone lacks the contracted axes).
#[cube]
fn mma_steps<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &Tile<E>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
) {
    let operands = comptime!(Space::merge(&[
        &acc.place.space,
        &lhs.place.space,
        &rhs.place.space
    ]));
    let space = witnessed_space(operands, acc, lhs, rhs);
    let walk = Region::rooted(
        &space,
        comptime!(acc.place.levels.clone()),
        comptime!(acc.place.depth),
    )
    .walk();
    for region in walk.unrolled() {
        let mut acc_region = acc.at(&region);
        mma_leaf::<E, Lhs, Rhs>(&mut acc_region, &lhs.at(&region), &rhs.at(&region));
    }
}

/// [`Descent::Cells`]: each operand's fragments loaded once, then every cell of the grid.
#[cube]
fn mma_cells<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &Tile<E>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    #[comptime] instruction: Instruction,
) {
    let lhs = PlanePartition::<Lhs>::operand(lhs, acc, instruction);
    let rhs = PlanePartition::<Rhs>::operand(rhs, acc, instruction);
    for cell in acc.walk().unrolled() {
        let mut acc_cell = acc.at(&cell);
        mma_leaf::<E, Lhs, Rhs>(&mut acc_cell, &lhs.at(&cell), &rhs.at(&cell));
    }
}

/// [`Descent::Here`]: `acc += lhs · rhs` into the one fragment or block `acc` holds, or the
/// memory window it is.
#[cube]
fn mma_here<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
) {
    let space = comptime!(acc.place.space.clone());
    let tile_kind = &mut acc.kind;
    match tile_kind {
        TileKind::PlaneTile(t) => t.mma(lhs, rhs, space),
        TileKind::PlanePartition(p) => {
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs, space)
        }
        TileKind::Memory(g) => {
            let contraction = comptime!(g.contraction.unwrap_or_else(unstated_contraction));
            memory::contract::<E, Lhs, Rhs>(
                g,
                lhs,
                rhs,
                space,
                contraction.block,
                contraction.semiring,
            )
        }
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) | TileKind::Lines(_) => {
            panic!("mma: a procedural tile and the plane's units are not an accumulator sink")
        }
    }
}

/// The semiring a plane-resident accumulator was opened under.
#[cube]
fn held_semiring<E: Numeric>(acc: &Tile<E>) -> comptime_type!(Semiring) {
    match &acc.kind {
        TileKind::PlaneTile(t) => t.semiring(),
        TileKind::PlanePartition(p) => p.at(0usize, 0usize).semiring(),
        TileKind::Memory(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => {
            panic!("Tile::reset: only a plane-resident accumulator has an identity to return to")
        }
    }
}

/// How a contraction into a memory window runs, as stated with [`Tile::accumulating`].
#[cube]
fn contraction_of<E: Numeric>(acc: &Tile<E>) -> comptime_type!(Contraction) {
    match &acc.kind {
        TileKind::Memory(g) => comptime!(g.contraction.unwrap_or_else(unstated_contraction)),
        TileKind::PlaneTile(_)
        | TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => panic!("Tile::mm: only a memory window states its contraction"),
    }
}

fn unstated_contraction() -> Contraction {
    panic!(
        "Tile::mma: a memory window contracts under the register block and semiring it is stated \
         with; state them with Tile::accumulating(block, semiring)"
    )
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
                // The manual-mma instruction reads registers, where a scaled factor cannot land.
                lhs.refuse_factor("PlaneTile::Mma");
                rhs.refuse_factor("PlaneTile::Mma");
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
         `memory::memory` arm of `mma_leaf`"
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
