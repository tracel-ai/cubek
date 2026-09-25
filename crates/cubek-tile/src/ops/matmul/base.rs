//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile: the leaf dispatch ([`mma_leaf`]) on the
//! accumulator's form. The levels above the leaf are the kernel's own walk; nothing here
//! recurses.
//!
//! The [`Semiring`] states the accumulation's algebra once, at the call that runs the steps.

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use super::leaf::memory;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` at a final register-resident tile (a plane fragment or a register block):
    /// the identity, then [`mma`](Tile::mma). A memory accumulator states the block it runs
    /// under instead ([`mm_with`](Tile::mm_with)).
    ///
    /// Each operand carries whatever scales [`mul`](Tile::mul) gave it, read where its values
    /// are read; one carrying none reads as it lies and costs nothing.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        self.init_identity(comptime!(semiring.add()));
        self.mma(lhs, rhs, semiring);
    }

    /// `c += a · b` at a final register-resident tile. Folds onto whatever `c` holds; nothing
    /// here initializes it.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        mma_leaf(self, lhs, rhs, semiring)
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` at a final memory tile through the software instruction run under `config`:
    /// the leaf a kernel walking its own levels reaches, with the register block stated rather
    /// than read off the space. `c` owns each cell outright, so the block starts from the identity.
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

/// The leaf contraction `acc += lhs · rhs`, dispatched on the accumulator's form. Each factor
/// carries its own scales, or none.
#[cube]
pub fn mma_leaf<E: Numeric, Lhs: Numeric, Rhs: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    #[comptime] semiring: Semiring,
) {
    let space = comptime!(acc.place.space.clone());
    let tile_kind = &mut acc.kind;
    match tile_kind {
        TileKind::PlaneTile(t) => t.mma(lhs, rhs, space, semiring),
        // A partition that reaches a final tile carries exactly one tile; a wider one is
        // consumed earlier, at its partition level.
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma(lhs, rhs, space, semiring)
        }
        // A memory accumulator runs the software instruction under a register block the kernel
        // states; this dispatch has none to hand it.
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

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// Contract this plane tile, each factor times whatever scales it carries.
    ///
    /// A hardware instruction eats its operands' format, so a scaled factor reaches one through
    /// memory: [`CmmaData::mma`] lands it, unpacked and scaled, in the plane's own window and
    /// loads the fragment from there. The manual-mma form has no landing, so it refuses one.
    pub fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                let transposed = transposed_rhs(lhs, rhs);
                strided_2d(lhs, rhs, comptime!(out.clone()), transposed);
                hardware_semiring(semiring);
                d.mma(lhs, rhs, out)
            }
            PlaneTile::Mma(d) => {
                // The manual-mma instruction takes its operands from registers, where a scaled
                // factor has nowhere to land.
                lhs.refuse_factor("PlaneTile::Mma");
                rhs.refuse_factor("PlaneTile::Mma");
                flattened_k(lhs, rhs, out);
                hardware_semiring(semiring);
                d.mma(lhs, rhs)
            }
            PlaneTile::Registers(d) => {
                let folded = comptime!(d.fold > 1);
                strided_2d(lhs, rhs, comptime!(out.clone()), folded);
                d.mma(lhs, rhs, out, semiring)
            }
        }
    }
}

/// Asserts that the algebra is the one a hardware instruction implements: it multiplies and adds.
#[cube]
fn hardware_semiring(#[comptime] semiring: Semiring) {
    comptime!(assert!(
        semiring == Semiring::SUM_PROD,
        "mma: a hardware instruction contracts under the sum-product semiring alone, not \
         {semiring:?}; contract in memory or in a register block to fold under another"
    ));
}

/// Asserts that operands are not gathered and read as one matrix each. A fragment contracts over
/// one `k` edge, not one contracted *axis*: axes carried as one run flatten into an edge (as a
/// partitioned contraction does); what it cannot read is a contraction its axes give no edge for.
///
/// The rhs reads `(k, col)`, or `(col, k)` where a register block folds a step (`rhs_along_k`):
/// lined along the contraction, its matrix is the transpose.
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

/// Whether `rhs` is read col-major: a cmma fragment loaded that way, or a staged `(col, k)`
/// window, the transpose of the role's order, read as the same matrix along the contracted edge
/// ([`PlanePartition::store`], [`rhs_layout`](crate::ops::matmul::leaf::rhs_layout)), like a folded step.
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
        // The contracted axis is the lhs's trailing one, as the leaf reads it: an axis the output
        // lacks is not always contracted (a spanned leading axis is not).
        // A shared stage is the one memory a fragment reads as it lies; a gmem rhs never
        // answers as column-major.
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
