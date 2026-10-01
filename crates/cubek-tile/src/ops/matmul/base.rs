//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile: the leaf dispatch ([`mma_leaf`]).

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use super::leaf::memory;
use crate::tile::base::witnessed_space;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` into a plane-resident accumulator, under the semiring it was opened with.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) {
        self.reset();
        self.mma(lhs, rhs);
    }

    /// `c += a · b` into a plane-resident accumulator, folding onto what it holds.
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
    /// `c = a · b` at a final memory tile through the software instruction run under `config`.
    pub fn mm_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        let init_from = self.request_init_from(comptime!(InitFrom::Identity));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => self.init_identity(comptime!(semiring.add())),
        }
        self.mma_with(lhs, rhs, config, semiring);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// `c += a · b` at a final memory tile through the software instruction run under `config`.
    pub fn mma_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        let space = comptime!(self.place.space.clone());
        match &mut self.kind {
            TileKind::Memory(g) => {
                memory::contract::<Acc, Lhs, Rhs>(g, lhs, rhs, space, config, semiring)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => panic!(
                "Tile::mma_with: the software instruction contracts into a memory accumulator; a \
                 register accumulator carries its own block (Tile::block_accumulator)"
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
    let fragments = fragments_held(&grid);
    if comptime!(fragments > 1) {
        mma_grid::<E, Lhs, Rhs>(&grid, lhs, rhs);
    } else {
        mma_fragment::<E, Lhs, Rhs>(acc, lhs, rhs);
    }
}

/// `acc += lhs · rhs` over a grid of fragments, as a kernel walks it. A level below that walks the
/// contraction is walked; at the level that only moves the grid's cells, each operand's fragments
/// are loaded once and contracted cell by cell.
#[cube]
fn mma_grid<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &Tile<E>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
) {
    let walks_contraction = comptime!(walks_contraction(
        &acc.place,
        &lhs.place.space,
        &rhs.place.space
    ));
    if walks_contraction {
        // Walked over the whole contraction's box: `acc`'s alone lacks the contracted axes.
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
    } else {
        let instruction = grid_instruction(acc);
        let lhs = fragments_of::<Lhs, E>(lhs, acc, instruction);
        let rhs = fragments_of::<Rhs, E>(rhs, acc, instruction);
        for cell in acc.walk().unrolled() {
            let mut acc_cell = acc.at(&cell);
            mma_leaf::<E, Lhs, Rhs>(&mut acc_cell, &lhs.at(&cell), &rhs.at(&cell));
        }
    }
}

/// `acc += lhs · rhs` into the one fragment or block `acc` holds.
#[cube]
fn mma_fragment<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
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
        TileKind::Memory(_) => panic!(
            "mma_leaf: a Gmem/Smem accumulator contracts through the software instruction, which \
             runs under a register block; state it with Tile::mma_with(lhs, rhs, config, semiring)"
        ),
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) | TileKind::Lines(_) => {
            panic!("mma: a procedural tile and the plane's units are not an accumulator sink")
        }
    }
}

/// How many fragments `acc` holds: a plane-resident grid's count, one for anything else.
#[cube]
fn fragments_held<E: Numeric>(acc: &Tile<E>) -> comptime_type!(usize) {
    match &acc.kind {
        TileKind::PlanePartition(p) => comptime!(p.m_tiles * p.n_tiles),
        TileKind::PlaneTile(_)
        | TileKind::Memory(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => comptime!(1usize),
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
        | TileKind::Lines(_) => panic!(
            "Tile::mma: only a plane-resident accumulator remembers its semiring; a memory tile \
             contracts through Tile::mma_with"
        ),
    }
}

/// The instruction a grid of fragments contracts through.
#[cube]
fn grid_instruction<E: Numeric>(acc: &Tile<E>) -> comptime_type!(Instruction) {
    match &acc.kind {
        TileKind::PlanePartition(p) => p.instruction(),
        TileKind::PlaneTile(_)
        | TileKind::Memory(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => panic!("mma: only a plane-resident grid holds several fragments"),
    }
}

/// `src` as fragments contracting into `acc`: loaded once from memory, or as it is if it already
/// holds them.
#[cube]
fn fragments_of<T: Numeric, E: Numeric>(
    src: &Tile<T>,
    acc: &Tile<E>,
    #[comptime] instruction: Instruction,
) -> Tile<T> {
    match &src.kind {
        TileKind::PlanePartition(_) | TileKind::PlaneTile(_) => src.clone(),
        TileKind::Memory(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_)
        | TileKind::Lines(_) => PlanePartition::<T>::operand(src, acc, instruction),
    }
}

/// Whether the level below `acc` walks an axis the operands contract, rather than only moving
/// `acc`'s own cells.
fn walks_contraction(acc: &Placement, lhs: &Space, rhs: &Space) -> bool {
    let below = acc.below().first().unwrap_or_else(|| {
        panic!("mma: a grid of fragments contracts at a level with its cells below it")
    });
    let operands = Space::merge(&[lhs, rhs]);
    below
        .axes()
        .iter()
        .any(|&axis| operands.contains(axis) && !acc.space.contains(axis))
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
