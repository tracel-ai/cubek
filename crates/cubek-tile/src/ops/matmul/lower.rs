//! `c.mm(a, b)` and `c.mma(a, b)` at a final tile: the leaf dispatch ([`mma_leaf`]) on the
//! accumulator's form. The levels above the leaf are the kernel's own walk; nothing here
//! recurses.
//!
//! The [`Semiring`] states the accumulation's algebra once, at the call that runs the steps.

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
        comptime!(assert!(
            self.space.is_final(),
            "Tile::mma: the leaf contracts a final tile; walk the levels above it first"
        ));
        mma_leaf(self, lhs, rhs, semiring)
    }

    /// `c = (a ⊗ s) · b`, or `c = a · (b ⊗ s)`: [`mm`](Tile::mm) with one operand scaled by a
    /// real operand.
    ///
    /// The scales are an operand like any other, and the arithmetic that folds them in is this
    /// verb: nothing decodes behind a read.
    ///
    /// **Which operand it scales is not stated**: the scales' own axes say it
    /// ([`ScaleSide`](crate::ScaleSide)). A scale spanning the output's columns is a fact about
    /// the rhs's columns; anything else scales the lhs. Both are the same sum of terms, so one
    /// verb serves both, folding once per `(row, k)` or once per `(col, k)`.
    ///
    /// `s` resolves at whatever granularity its axes give it, and cannot vary over an axis it does
    /// not address. The block is an axis of the problem, `(KB, KI)` or `(NB, NI)`, spelled with
    /// [`PhysicalAxisMap::disjoint`](crate::PhysicalAxisMap::disjoint) on the values while the
    /// scales leave the position inside it unmapped, so no line can straddle a block whatever
    /// width it is served at. A scales operand that divides instead is refused.
    pub fn mm_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Sequence<Tile<S>>,
        #[comptime] semiring: Semiring,
    ) {
        self.init_identity(comptime!(semiring.add()));
        self.mma_scaled(lhs, rhs, scales, semiring);
    }

    /// `c += (a ⊗ s) · b` (or its rhs twin): [`mma`](Tile::mma)'s scaled form.
    pub fn mma_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Sequence<Tile<S>>,
        #[comptime] semiring: Semiring,
    ) {
        comptime!(assert!(
            self.space.is_final(),
            "Tile::mma_scaled: the leaf contracts a final tile; walk the levels above it first"
        ));
        mma_leaf_scaled(self, lhs, rhs, scales, semiring)
    }

    /// The level's operation space: the merge of the operands' spaces, sized by whichever operand
    /// [`witnesses`](Tile::witnesses) each [`Dynamic`](crate::Extent) axis. The output contributes
    /// no axis beyond `lhs ∪ rhs`. What a kernel walks at a level ([`Walk::over`]).
    ///
    /// The accumulator is asked for sizes all the same, and first: spanning an axis and being able
    /// to state its size are different things (a gathered operand's bound is the receptive field
    /// its axes reach over, so it answers for neither), and an axis the output spans is one it
    /// writes, so its bound is the extent the walk must cover.
    pub fn op_space<Lhs: Numeric, Rhs: Numeric>(&self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>) -> Space {
        let merged = comptime!({
            let merged = Space::merge(&[&lhs.space, &rhs.space]);
            assert!(
                self.space.axes().all(|axis| merged.contains(axis)),
                "Tile::mma: the output spans an axis neither operand does, so the walk would never \
                 step it and every region would write the same slice"
            );
            merged
        });
        witnessed_space(merged, self, lhs, rhs)
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// `c = a · b` at a final memory tile through the software instruction run under `config`:
    /// the leaf a kernel that walks its own levels reaches, stated with the register block it
    /// runs rather than read off the space. `c` owns each cell outright here, so the block
    /// starts from the identity and never reads `c` back. The sum runs at this tile's element;
    /// [`mm_with_acc`](Self::mm_with_acc) carries it at a wider one.
    pub fn mm_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        self.mm_with_acc::<Acc, Lhs, Rhs>(lhs, rhs, config, semiring);
    }

    /// [`mm_with`](Self::mm_with) with the sum carried in `EA` and cast to this tile's element
    /// as it is written: what a half-precision output needs, since a cell summed in its own
    /// element stops growing once a product falls under half its spacing — an `f16` sum of
    /// values near one goes no further than 2048. The register path states the same choice at
    /// [`block_accumulator`](Self::block_accumulator).
    pub fn mm_with_acc<EA: Numeric, Lhs: Numeric, Rhs: Numeric>(
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
        self.mma_with_acc::<EA, Lhs, Rhs>(lhs, rhs, config, semiring);
        self.request_init_from(comptime!(InitFrom::Cell));
    }

    /// `c = a · b` over this tile's last level, whose regions step the contraction: one register
    /// block seeded once spans the walk and commits once, so the sum crosses the steps without
    /// touching the cells, and the lanes fold once. The sum is carried in `EA`.
    ///
    /// What a layout that deals a lane interleaved chunks of `K` needs. A leaf per chunk
    /// ([`mma_with_acc`](Self::mma_with_acc) on each region) folds across the lanes and writes
    /// the cell per chunk, and a half-precision cell rounds at every one of them.
    ///
    /// The level may distribute an accumulator axis — each lane its own rows — but not step
    /// one: a walked accumulator axis would move the cells under the block, and a distributed
    /// one handing an instance several tiles would too.
    pub fn mm_steps_with_acc<EA: Numeric, Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        comptime!(assert!(
            !self.space.is_final(),
            "Tile::mm_steps_with_acc: the tile has no level left to step; a final tile contracts \
             through mm_with"
        ));
        comptime!({
            for p in 0..self.space.rank() {
                let axis = self.space.axis_at(p);
                let tiles = self.space.count(axis);
                let per_instance = match self.space.partitioner().distribution(axis) {
                    Distribution::Sequential => tiles,
                    Distribution::Spatial { coverage, .. } => match coverage {
                        Coverage::Instances(n) => tiles / n.max(1),
                        Coverage::TilesEach(t) => t,
                    },
                };
                assert!(
                    per_instance == 1,
                    "Tile::mm_steps_with_acc: the level steps the accumulator's {axis:?} axis, so \
                     its regions do not share their cells; only a contracted axis may be stepped"
                );
            }
        });
        let walk = Walk::over(self.op_space(lhs, rhs));
        // Every step shares the cells of the first: the level moves the contraction alone.
        let mut cells = self.at(&walk.region(0));
        let init_from = cells.request_init_from(comptime!(InitFrom::Identity));
        match comptime!(init_from) {
            InitFrom::Identity => {}
            InitFrom::Cell => cells.init_identity(comptime!(semiring.add())),
        }
        let space = comptime!(cells.space.clone());
        match &mut cells.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => contract::memory_steps::<EA, Lhs, Rhs, Acc>(
                g, lhs, rhs, walk, space, config, semiring,
            ),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => panic!(
                "Tile::mm_steps_with_acc: the software instruction contracts into a memory \
                 accumulator; a register accumulator carries its own block (Tile::block_accumulator)"
            ),
        }
    }

    /// `c += a · b` at a final memory tile through the software instruction run under `config`.
    pub fn mma_with<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        self.mma_with_acc::<Acc, Lhs, Rhs>(lhs, rhs, config, semiring);
    }

    /// [`mma_with`](Self::mma_with) with the sum carried in `EA`: the cells are read widened
    /// and written narrowed, and every partial on the way — a lane's, a group's, the plane's —
    /// is combined at `EA`.
    pub fn mma_with_acc<EA: Numeric, Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        comptime!(assert!(
            self.space.is_final(),
            "Tile::mma_with: the software instruction runs on a final tile; walk the levels above \
             it first"
        ));
        let space = comptime!(self.space.clone());
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                contract::memory::<EA, Lhs, Rhs, Acc>(g, lhs, rhs, space, config, semiring)
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

    /// `c += (a ⊗ s) · b`, or its rhs twin, at a final memory tile through the software
    /// instruction run under `config`.
    pub fn mma_scaled_with<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Sequence<Tile<S>>,
        #[comptime] config: RegisterBlock,
        #[comptime] semiring: Semiring,
    ) {
        comptime!(assert!(
            self.space.is_final(),
            "Tile::mma_scaled_with: the software instruction runs on a final tile; walk the \
             levels above it first"
        ));
        let space = comptime!(self.space.clone());
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                contract::memory_scaled::<Acc, Lhs, Rhs, S, Acc>(
                    g, lhs, rhs, scales, space, config, semiring,
                )
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => panic!(
                "Tile::mma_scaled_with: the software instruction contracts into a memory \
                 accumulator"
            ),
        }
    }
}

/// The leaf contraction `acc += lhs · rhs`, dispatched on the accumulator's form.
#[cube]
pub fn mma_leaf<E: Numeric, EL: Numeric, ER: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
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

/// [`mma_leaf`] with one operand scaled, on a register-block accumulator, the form whose step
/// has a scale to apply. A fragment accumulator contracts through a hardware instruction that
/// takes two operands and no scales, so a scaled contraction there is a different instruction,
/// not this one under a flag.
#[cube]
pub(crate) fn mma_leaf_scaled<E: Numeric, EL: Numeric, ER: Numeric, S: Numeric>(
    acc: &mut Tile<E>,
    lhs: &Tile<EL>,
    rhs: &Tile<ER>,
    scales: &Sequence<Tile<S>>,
    #[comptime] semiring: Semiring,
) {
    let space = comptime!(acc.space.clone());
    let tile_kind = &mut acc.tile_kind;
    match tile_kind {
        // A promoted register accumulator: the partials stay in `E` across the whole walk, which
        // is the form a decode gemv wants.
        TileKind::PlaneTile(t) => t.mma_scaled(lhs, rhs, scales, space, semiring),
        TileKind::PlanePartition(p) => {
            comptime!(assert!(
                p.m_tiles == 1 && p.n_tiles == 1,
                "mma_leaf_scaled: a multi-tile partition must be contracted at its partition level"
            ));
            let mut t = p.at(0usize, 0usize);
            t.mma_scaled(lhs, rhs, scales, space, semiring)
        }
        TileKind::Gmem(_) | TileKind::Smem(_) => panic!(
            "mma_leaf_scaled: a Gmem/Smem accumulator contracts through the software \
             instruction, which runs under a register block; state it with \
             Tile::mma_scaled_with(lhs, rhs, scales, config, semiring)"
        ),
        TileKind::TmaGmem(_) => panic!("mma_scaled: a tma source is not an accumulator sink"),
        TileKind::Procedural(_) => {
            panic!("mma_scaled: a procedural tile is not an accumulator sink")
        }
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// Contract this plane tile.
    pub fn mma<EL: Numeric, ER: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            PlaneTile::Cmma(d) => {
                strided_2d(lhs, rhs, out);
                hardware_semiring(semiring);
                d.mma(lhs, rhs)
            }
            PlaneTile::Mma(d) => {
                flattened_k(lhs, rhs, out);
                hardware_semiring(semiring);
                d.mma(lhs, rhs)
            }
            PlaneTile::Register(d) => {
                strided_2d(lhs, rhs, comptime!(out.clone()));
                d.mma(lhs, rhs, out, semiring)
            }
        }
    }
}

#[cube]
impl<E: Numeric> PlaneTile<E> {
    /// [`mma`](PlaneTile::mma) with one operand scaled by a real operand. Only the register form:
    /// a hardware instruction eats its operands' format whole, so a scale there routes to the
    /// *fragment* rather than to a view, which is a different instruction.
    pub fn mma_scaled<EL: Numeric, ER: Numeric, ES: Numeric>(
        &mut self,
        lhs: &Tile<EL>,
        rhs: &Tile<ER>,
        scales: &Sequence<Tile<ES>>,
        #[comptime] out: Space,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            PlaneTile::Register(d) => {
                strided_2d(lhs, rhs, comptime!(out.clone()));
                d.mma_scaled(lhs, rhs, scales, out, semiring)
            }
            PlaneTile::Cmma(_) | PlaneTile::Mma(_) => panic!(
                "mma_scaled: a hardware instruction eats its operands' format, so a scaled                  contraction on a fragment accumulator needs a scaled hardware instruction"
            ),
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
/// give no edge for.
#[cube]
fn strided_2d<EL: Numeric, ER: Numeric>(lhs: &Tile<EL>, rhs: &Tile<ER>, #[comptime] out: Space) {
    let lhs_gathered = lhs.gathered();
    let rhs_gathered = rhs.gathered();
    let flat = comptime!({
        let kc = Space::merge(&[&lhs.space, &rhs.space]).contracted_extent(&out);
        let axes = MatrixAxes::accumulator(&out, &lhs.space);
        MatrixAxes::find(&lhs.space, axes.rows(&out), kc).is_some()
            && MatrixAxes::find(&rhs.space, kc, axes.cols(&out)).is_some()
    });
    comptime!(assert!(
        !lhs_gathered && !rhs_gathered && flat,
        "mma: a cmma or plane-register fragment reads one `k` edge off a directly addressed \
         operand; a gather, or a contraction these axes give no edge for, needs the manual-mma \
         leaf, or an unpromoted Gmem/Smem accumulator, whose software instruction is the \
         `contract::memory` arm of `mma_leaf`"
    ));
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
