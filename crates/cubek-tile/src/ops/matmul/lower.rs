//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile: the leaf dispatch ([`mma_leaf`]) on the
//! accumulator's form. The levels above the leaf are the kernel's own walk; nothing here
//! recurses.
//!
//! The [`Semiring`] states the accumulation's algebra once, at the call that runs the steps.

use cubecl::cmma::MatrixLayout;
use cubecl::prelude::*;

use crate::instruction::registers::contract;
use crate::*;

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` at a final register-resident tile (a plane fragment or a register block):
    /// the identity, then [`mma`](Tile::mma). A memory accumulator states the block it runs
    /// under instead ([`mm_with`](Tile::mm_with)).
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        self.mm_scaled(&lhs.plain(), &rhs.plain(), semiring);
    }

    /// `c += a · b` at a final register-resident tile. Folds onto whatever `c` holds; nothing
    /// here initializes it.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        self.mma_scaled(&lhs.plain(), &rhs.plain(), semiring);
    }

    /// [`mm`](Tile::mm) over factors that carry their own scales: `c = (a ⊗ s) · b` is
    /// `c.mm_scaled(&a.scaled(&s), &b.plain(), semiring)`.
    ///
    /// Which factor a scale multiplies is where the kernel wrote it, and how deep the scales go
    /// is how many times it said `scaled`. Nothing infers a side and nothing counts levels; a
    /// factor carrying none reads as its values alone, which is what [`mm`](Tile::mm) hands it.
    pub fn mm_scaled<Lhs: Numeric, LS: Numeric, Rhs: Numeric, RS: Numeric>(
        &mut self,
        lhs: &Scaled<Lhs, LS>,
        rhs: &Scaled<Rhs, RS>,
        #[comptime] semiring: Semiring,
    ) {
        self.init_identity(comptime!(semiring.add()));
        self.mma_scaled(lhs, rhs, semiring);
    }

    /// [`mma`](Tile::mma) over factors that carry their own scales.
    pub fn mma_scaled<Lhs: Numeric, LS: Numeric, Rhs: Numeric, RS: Numeric>(
        &mut self,
        lhs: &Scaled<Lhs, LS>,
        rhs: &Scaled<Rhs, RS>,
        #[comptime] semiring: Semiring,
    ) {
        mma_leaf(self, lhs, rhs, semiring)
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` at a final memory tile through the software instruction run under `config`:
    /// the leaf a kernel that walks its own levels reaches, stated with the register block it
    /// runs rather than read off the space. `c` owns each cell outright here, so the block
    /// starts from the identity and never reads `c` back.
    pub fn mm_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        self.mm_scaled_with(&lhs.plain(), &rhs.plain(), config, semiring);
    }

    /// `c += a · b` at a final memory tile through the software instruction run under `config`.
    pub fn mma_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        self.mma_scaled_with(&lhs.plain(), &rhs.plain(), config, semiring);
    }

    /// [`mm_with`](Tile::mm_with) over factors that carry their own scales.
    pub fn mm_scaled_with<Lhs: Numeric, LS: Numeric, Rhs: Numeric, RS: Numeric>(
        &mut self,
        lhs: &Scaled<Lhs, LS>,
        rhs: &Scaled<Rhs, RS>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        let init_from = self.request_init_from(comptime!(InitFrom::Identity));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => self.init_identity(comptime!(semiring.add())),
        }
        self.mma_scaled_with(lhs, rhs, config, semiring);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// [`mma_with`](Tile::mma_with) over factors that carry their own scales.
    pub fn mma_scaled_with<Lhs: Numeric, LS: Numeric, Rhs: Numeric, RS: Numeric>(
        &mut self,
        lhs: &Scaled<Lhs, LS>,
        rhs: &Scaled<Rhs, RS>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        let space = comptime!(self.space.clone());
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                contract::memory::<Acc, Lhs, LS, Rhs, RS>(g, lhs, rhs, space, config, semiring)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => panic!(
                "Tile::mma_with: the software instruction contracts into a memory accumulator; a \
                 register accumulator carries its own block (Tile::block_accumulator)"
            ),
        }
    }
}

/// The leaf contraction `acc += lhs · rhs`, dispatched on the accumulator's form. Each factor
/// carries its own scales, or none.
#[cube]
pub fn mma_leaf<E: Numeric, Lhs: Numeric, LS: Numeric, Rhs: Numeric, RS: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Scaled<Lhs, LS>,
    rhs: &Scaled<Rhs, RS>,
    #[comptime] semiring: Semiring,
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
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
        TileKind::Gmem(_) | TileKind::Smem(_) => panic!(
            "mma_leaf: a Gmem/Smem accumulator contracts through the software instruction, which \
             runs under a register block; state it with Tile::mma_with(lhs, rhs, config, semiring)"
        ),
        TileKind::TmaGmem(_) => panic!("mma: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) => panic!("mma: a procedural tile is not an accumulator sink"),
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// Contract this plane tile, each factor times whatever scales it carries.
    ///
    /// A hardware instruction eats its operands' format, so a scaled factor reaches one through
    /// memory: [`CmmaData::mma_scaled`] lands it, unpacked and scaled, in the plane's own window
    /// and loads the fragment from there. The manual-mma form takes its operands from registers
    /// and has no such landing, so it refuses one.
    pub fn mma<EL: Numeric, LS: Numeric, ER: Numeric, RS: Numeric>(
        &mut self,
        lhs: &Scaled<EL, LS>,
        rhs: &Scaled<ER, RS>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        let lhs_values = lhs.values();
        let rhs_values = rhs.values();
        let lhs_count = lhs.levels().len();
        let rhs_count = rhs.levels().len();
        let scaled = comptime!(lhs_count > 0 || rhs_count > 0);
        match self {
            PlaneTile::Cmma(d) => {
                let transposed = transposed_rhs(&lhs_values, &rhs_values);
                strided_2d(&lhs_values, &rhs_values, comptime!(out.clone()), transposed);
                hardware_semiring(semiring);
                if comptime!(scaled) {
                    d.mma_scaled(lhs, rhs, out)
                } else {
                    d.mma(&lhs_values, &rhs_values)
                }
            }
            PlaneTile::Mma(d) => {
                comptime!(assert!(
                    !scaled,
                    "mma: the manual-mma instruction takes its operands from registers, where a \
                     scaled factor has nowhere to land; open a cmma accumulator"
                ));
                flattened_k(&lhs_values, &rhs_values, out);
                hardware_semiring(semiring);
                d.mma(&lhs_values, &rhs_values)
            }
            PlaneTile::Register(d) => {
                let folded = comptime!(d.fold > 1);
                strided_2d(&lhs_values, &rhs_values, comptime!(out.clone()), folded);
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
/// one `k` edge, which is not one contracted *axis*: axes carried as one run flatten into an edge,
/// and a partitioned contraction is exactly that. What it cannot read is a contraction its axes
/// give no edge for. The rhs reads `(k, col)`, or `(col, k)` where a register block folds a step
/// (`rhs_along_k`): lined along the contraction, its matrix is the transpose.
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
        let kc = Space::merge(&[&lhs.space, &rhs.space]).contracted_extent(&out);
        let axes = MatrixAxes::accumulator(&out, &lhs.space);
        let cols = axes.cols(&out);
        let rhs_matrix = if rhs_along_k {
            MatrixAxes::find(&rhs.space, cols, kc)
        } else {
            MatrixAxes::find(&rhs.space, kc, cols)
        };
        MatrixAxes::find(&lhs.space, axes.rows(&out), kc).is_some() && rhs_matrix.is_some()
    });
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered && flat,
        "mma: a cmma or plane-register fragment reads one `k` edge off a directly addressed \
         operand; a gather, or a contraction these axes give no edge for, needs the manual-mma \
         leaf, or an unpromoted Gmem/Smem accumulator, whose software instruction is the \
         `contract::memory` arm of `mma_leaf`"
    ));
}

/// Whether `rhs` is read col-major: a cmma fragment loaded that way, or a staged `(col, k)`
/// window, the transpose of the role's own order, which the leaf reads as the same matrix
/// ([`PlanePartition::store`], [`rhs_layout`](crate::instruction::rhs_layout)) and which is
/// therefore the edge the contraction runs along, as it is for a folded register step.
#[cube]
fn transposed_rhs<EL: Numeric, ER: Numeric>(
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
) -> comptime_type!(bool) {
    match &rhs.tile_kind {
        TileKind::PlaneTile(t) => match t {
            PlaneTile::Cmma(d) => comptime!(d.layout == MatrixLayout::ColMajor),
            PlaneTile::Mma(_) | PlaneTile::Register(_) => comptime!(false),
        },
        // The contracted axis is the lhs's trailing one, as the leaf reads it: an axis the output
        // lacks is not always contracted (a spanned leading axis is not).
        TileKind::Smem(_) => comptime!(
            crate::instruction::rhs_layout(&rhs.space, lhs.space.axis_at(lhs.space.rank() - 1))
                == MatrixLayout::ColMajor
        ),
        TileKind::Gmem(_)
        | TileKind::PlanePartition(_)
        | TileKind::TmaGmem(_)
        | TileKind::Procedural(_) => comptime!(false),
    }
}

/// Asserts that operands contract their shared axes in the same order.
#[cube]
fn flattened_k<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    comptime!(assert!(
        Space::contraction_agrees(&lhs.space, &rhs.space, &out),
        "mma: the operands list their contracted axes in different orders ({:?} against {:?}), \
         so their `k` edges do not line up",
        lhs.space.contracting(&out),
        rhs.space.contracting(&out)
    ));
}
