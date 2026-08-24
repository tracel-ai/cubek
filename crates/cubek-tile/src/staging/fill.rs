//! Building and filling the staging slots of a ring: the operand-deducing [`SlotPlan`], the
//! ring constructors ([`Ring::binary`] / [`Ring::unary`]), the fixed/streamed split
//! ([`fill_fixed`](Staging::fill_fixed) / [`fill_streamed`](Staging::fill_streamed)), and the
//! closure-driven [`fill`](Staging::fill) / [`consume`](Staging::consume) with their hand-written
//! expands.
//!
//! `fill`/`consume` are hand-written expand methods because a `Drop` guard can't emit a barrier
//! op in cubecl and `#[cube]` rejects `impl Trait` args.

use core::option::Option;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// One operand's comptime facts, read off its [`Tile`] before any slot is built.
pub(crate) struct SlotOperand<'a> {
    residence: Residence,
    source: StageSource,
    space: &'a Space,
}

impl<'a> SlotOperand<'a> {
    pub(crate) fn new(residence: Residence, source: StageSource, space: &'a Space) -> Self {
        SlotOperand {
            residence,
            source,
            space,
        }
    }
}

/// What every slot of one ring decides at comptime: how it rendezvouses, and per operand where it
/// lives and when it is filled. Deduced once from the operands, so a ring's slots agree by
/// construction rather than by each re-deriving the same answers off the same tiles.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct SlotPlan {
    operands: Vec<OperandPlan>,
    sync: Sync,
    collective_full: bool,
}

impl SlotPlan {
    pub(crate) fn new(operands: &[SlotOperand], op_space: &Space) -> SlotPlan {
        // Admission first: the rest of the plan reads an operand as a fill source, which a
        // resident fragment can only be if something asked to materialize it here.
        assert!(
            slot_admits_operands(operands),
            "Staging: an operand already materialized as a plane fragment has no source to \
             materialize from, so it can only be read where it lies (Residence::InPlace); \
             materialize it at a level above instead"
        );
        let residences: Vec<_> = operands.iter().map(|op| op.residence).collect();
        if !compatible_slot_residences(&residences) {
            if operands
                .iter()
                .any(|op| op.source.delivery() == Some(Delivery::Procedural))
            {
                panic!(
                    "Staging: a procedural recipe stages in Smem and cannot share a slot with a Plane operand"
                );
            } else {
                panic!("Staging: Plane and Smem operands cannot share a slot");
            }
        }
        // Fix an operand only when its window is genuinely invariant across the walk. A barrier
        // pipeline arrives `full` once per fill, so a TMA pair keeps the joint per-region fill;
        // splitting an invariant out would corrupt its phase. A dynamic level can't decide
        // invariance at comptime. Both fall back to streaming.
        let can_fix_invariants = op_space.is_static()
            && !operands
                .iter()
                .filter_map(|op| op.source.delivery())
                .any(|delivery| delivery.is_tma());
        let planned_operands = operands
            .iter()
            .map(|op| {
                // An in-place operand is never windowed: no buffer is allocated for it, so there
                // is nothing to fill and nothing whose invariance could save a fill.
                let payload = if op.residence == Residence::InPlace {
                    SlotPayload::AtRegion
                } else if can_fix_invariants && op_space.walk_invariant(op.space) {
                    SlotPayload::Windowed(WindowMode::Fixed)
                } else {
                    SlotPayload::Windowed(WindowMode::Streamed)
                };
                OperandPlan::new(payload, op.residence, op.source)
            })
            .collect();
        let (sync, collective_full) = slot_sync(&rendezvous_deliveries(operands));
        SlotPlan {
            operands: planned_operands,
            sync,
            collective_full,
        }
    }

    /// What ring slot `slot` holds for operand `operand`. The first slot owns every payload;
    /// a later slot reuses the first slot's buffer for whatever the walk never rewrites.
    pub(crate) fn operand_plan(&self, operand: usize, slot: usize) -> OperandPlan {
        let mut plan = self.operands[operand];
        if slot != FIRST_SLOT {
            plan.payload = plan.payload.in_later_slot(plan.residence);
        }
        plan
    }

    /// Whether slot `slot` takes the first slot's buffer for `operand` instead of allocating one.
    pub(crate) fn reuses_first_buffer(&self, operand: usize, slot: usize) -> bool {
        self.operand_plan(operand, slot)
            .payload
            .reuses_first_buffer()
    }

    pub(crate) fn sync(&self) -> Sync {
        self.sync
    }

    pub(crate) fn collective_full(&self) -> bool {
        self.collective_full
    }
}

/// The slot-wide sync and whether all units publish its full barrier, derived only from operands
/// that physically materialize. A slot materializing nothing has nothing to synchronize.
fn slot_sync(deliveries: &[Delivery]) -> (Sync, bool) {
    let sync = if deliveries.is_empty() {
        Sync::Solo
    } else {
        Sync::for_deliveries(deliveries)
    };
    (sync, Sync::collective_full(deliveries))
}

/// Only operands that physically materialize join the slot's rendezvous: an in-place operand is
/// read where it lies, moving no bytes and needing no barrier.
fn rendezvous_deliveries(operands: &[SlotOperand]) -> Vec<Delivery> {
    operands
        .iter()
        .filter(|op| op.residence == Residence::Smem)
        .filter_map(|op| op.source.delivery())
        .collect()
}

/// Whether two operands can share one slot. Stated per pair rather than by counting kinds, so a
/// new [`Residence`] has to say how it meets each of the others instead of defaulting to
/// compatible.
fn compatible_residence_pair(a: Residence, b: Residence) -> bool {
    match (a, b) {
        // A plane partition is private to its unit and assumes a solo fill; a shared-memory stage
        // selects a slot-wide Cube or Barrier pipeline. One slot cannot rendezvous both ways.
        (Residence::Register, Residence::Smem) | (Residence::Smem, Residence::Register) => false,
        (Residence::Register, Residence::Register | Residence::InPlace) => true,
        (Residence::Smem, Residence::Smem | Residence::InPlace) => true,
        (Residence::InPlace, Residence::Register | Residence::Smem | Residence::InPlace) => true,
    }
}

/// Whether every pair of `residences` can share one slot. The self-pair is included, so every
/// residence must be compatible with itself.
fn compatible_slot_residences(residences: &[Residence]) -> bool {
    residences.iter().enumerate().all(|(i, &a)| {
        residences[i..]
            .iter()
            .all(|&b| compatible_residence_pair(a, b))
    })
}

/// A slot fills every operand it materializes from a source, and an operand a level above already
/// materialized into plane-private registers has none: its cells are registers, which no transport
/// reads. Such an operand can only be read where it lies ([`InPlace`](Residence::InPlace), so
/// [`AtRegion`](SlotPayload::AtRegion)), and reaching here with any other residence means something
/// asked to re-materialize a fragment.
fn slot_admits_operands(operands: &[SlotOperand]) -> bool {
    operands.iter().all(|op| match op.source {
        StageSource::ResidentFragment => op.residence == Residence::InPlace,
        StageSource::Transport(_) => true,
    })
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Ring<(Tile<Lhs>, Tile<Rhs>)> {
    /// Build the `depth` slots staging the operands `lhs`/`rhs`. A [`Residence::Register`] operand
    /// stages into plane-private tile partitions ([`Solo`](Sync::Solo)); [`Smem`](Residence::Smem)
    /// takes fresh shared memory, with [`Sync`] deduced from the operands' delivery; an
    /// [`InPlace`](Residence::InPlace) one takes no buffer at all.
    ///
    /// Slot 0 allocates every payload. A later slot allocates only what the walk refills: an
    /// operand whose window the walk leaves fixed is filled once and never rewritten, so one
    /// buffer serves the whole ring ([`Reused`](WindowMode::Reused)).
    pub fn binary(
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
        #[comptime] depth: usize,
    ) -> Ring<(Tile<Lhs>, Tile<Rhs>)> {
        let lhs_residence = lhs.residence(comptime!(&out));
        let rhs_residence = rhs.residence(comptime!(&out));
        let lhs_source = lhs.stage_source();
        let rhs_source = rhs.stage_source();
        let plan = comptime!(SlotPlan::new(
            &[
                SlotOperand::new(lhs_residence, lhs_source, &lhs.space),
                SlotOperand::new(rhs_residence, rhs_source, &rhs.space),
            ],
            &op_space,
        ));

        let mut slots = Sequence::<Staging<(Tile<Lhs>, Tile<Rhs>)>>::new();
        #[unroll]
        for slot in 0..depth {
            let staged_lhs = if comptime!(plan.reuses_first_buffer(LHS, slot)) {
                slots.index(FIRST_SLOT).data.0.clone()
            } else {
                stage_operand(lhs, comptime!(out.clone()), lhs_residence)
            };
            let staged_rhs = if comptime!(plan.reuses_first_buffer(RHS, slot)) {
                slots.index(FIRST_SLOT).data.1.clone()
            } else {
                stage_operand(rhs, comptime!(out.clone()), rhs_residence)
            };
            let staging = Staging::wrap(
                (staged_lhs, staged_rhs),
                Pipeline::new(comptime!(plan.sync()), comptime!(plan.collective_full())),
                comptime!(plan.operand_plan(LHS, slot)),
                comptime!(Option::Some(plan.operand_plan(RHS, slot))),
            );
            slots.push(staging);
        }
        Ring::wrap(slots, depth)
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Fill the fixed operand(s), those the walk leaves invariant, from `region`'s window.
    /// Their window never moves, so `region` is region 0 and this runs once, above the loop.
    /// A no-op when nothing is fixed, and when a later ring slot reuses the first's buffers.
    pub fn fill_fixed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let lhs_plan = self.plan(LHS);
        let rhs_plan = self.plan(RHS);
        let fixed_lhs = comptime!(lhs_plan.payload.is_fixed());
        let fixed_rhs = comptime!(rhs_plan.payload.is_fixed());
        if comptime!(fixed_lhs || fixed_rhs) {
            self.fill(|staged_operands, pipe| {
                if comptime!(fixed_lhs) {
                    pipe.fill(&mut staged_operands.0, &lhs.at(region));
                }
                if comptime!(fixed_rhs) {
                    pipe.fill(&mut staged_operands.1, &rhs.at(region));
                }
            });
        }
    }

    /// Fill the streamed operand(s), everything the walk moves, from `region`'s window. Runs per
    /// region inside the walk. The slot still rendezvouses when every operand is fixed: the
    /// pipeline's phase belongs to the slot, not to any one operand.
    pub fn fill_streamed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let lhs_plan = self.plan(LHS);
        let rhs_plan = self.plan(RHS);
        let stream_lhs = comptime!(lhs_plan.payload.is_streamed());
        let stream_rhs = comptime!(rhs_plan.payload.is_streamed());
        self.fill(|staged_operands, pipe| {
            if comptime!(stream_lhs) {
                pipe.fill(&mut staged_operands.0, &lhs.at(region));
            }
            if comptime!(stream_rhs) {
                pipe.fill(&mut staged_operands.1, &rhs.at(region));
            }
        });
    }
}

// `fill`/`consume` take closures so the body stays caller-defined (fill each buffer however, run the
// mma). They're provided for the `(Tile<Lhs>, Tile<Rhs>)` payload (not generic `T`): closure-parameter
// inference can't resolve the projection `&mut T::ExpandType` through a generic `T`, but resolves the
// concrete `TileExpand` fields of the pair directly.
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Producer: wait the slot is free, run `fill` over the staged buffers and the slot's
    /// [`Pipeline`], then publish. See [`StagingExpand::__expand_fill_method`].
    pub fn fill(&mut self, _fill: impl FnOnce(&mut (Tile<Lhs>, Tile<Rhs>), &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the two staged tiles to `compute`, then free the slot.
    /// Each tile's bytes and runtime map were stored together by the producer. A payload the slot
    /// filled is already this region's; an in-place one is the operand whole, and the caller
    /// selects the region out of it ([`read_operand`]).
    /// See [`StagingExpand::__expand_consume_method`].
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagingExpand<(Tile<Lhs>, Tile<Rhs>)> {
    pub fn __expand_fill_method<F>(&mut self, scope: &Scope, fill: F)
    where
        F: FnOnce(&Scope, &mut (TileExpand<Lhs>, TileExpand<Rhs>), &PipelineExpand),
    {
        self.__expand_acquire_write_method(scope);
        fill(scope, &mut self.data, &self.pipeline);
        self.__expand_release_write_method(scope);
    }

    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<Lhs>, &TileExpand<Rhs>),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data.0, &self.data.1);
        self.__expand_release_read_method(scope);
    }
}

#[cube]
impl<T: Numeric> Ring<Tile<T>> {
    /// Build the `depth` slots staging the sole operand `input`. See [`Ring::binary`] for how a
    /// later slot reuses the first slot's buffer.
    pub fn unary(
        input: &Tile<T>,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
        #[comptime] depth: usize,
    ) -> Ring<Tile<T>> {
        let residence = input.residence(comptime!(&out));
        let source = input.stage_source();
        let plan = comptime!(SlotPlan::new(
            &[SlotOperand::new(residence, source, &input.space)],
            &op_space,
        ));

        let mut slots = Sequence::<Staging<Tile<T>>>::new();
        #[unroll]
        for slot in 0..depth {
            let staged_input = if comptime!(plan.reuses_first_buffer(LHS, slot)) {
                slots.index(FIRST_SLOT).data.clone()
            } else {
                stage_operand(input, comptime!(out.clone()), residence)
            };
            let staging = Staging::wrap(
                staged_input,
                Pipeline::new(comptime!(plan.sync()), comptime!(plan.collective_full())),
                comptime!(plan.operand_plan(LHS, slot)),
                comptime!(Option::None),
            );
            slots.push(staging);
        }
        Ring::wrap(slots, depth)
    }
}

#[cube]
impl<T: Numeric> Staging<Tile<T>> {
    /// Fill the operand from `region`'s window if the walk leaves it fixed.
    pub fn fill_fixed(&mut self, input: &Tile<T>, region: &Region) {
        let plan = self.plan(LHS);
        let fixed = comptime!(plan.payload.is_fixed());
        if comptime!(fixed) {
            self.fill(|s, pipe| {
                pipe.fill(s, &input.at(region));
            });
        }
    }

    /// Fill the operand from `region`'s window if the walk moves it. The slot still rendezvouses
    /// otherwise: the pipeline's phase belongs to the slot, not to the operand.
    pub fn fill_streamed(&mut self, input: &Tile<T>, region: &Region) {
        let plan = self.plan(LHS);
        let stream = comptime!(plan.payload.is_streamed());
        self.fill(|s, pipe| {
            if comptime!(stream) {
                pipe.fill(s, &input.at(region));
            }
        });
    }
}

impl<T: Numeric> Staging<Tile<T>> {
    /// Producer: wait the slot is free, run `fill` over the staged buffer and the slot's
    /// [`Pipeline`], then publish.
    pub fn fill(&mut self, _fill: impl FnOnce(&mut Tile<T>, &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the staged tile to `compute`, then free the slot.
    /// Each tile's bytes and runtime map were stored together by the producer. A payload the slot
    /// filled is already this region's; an in-place one is the operand whole, and the caller
    /// selects the region out of it ([`read_operand`]).
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<T>)) {
        unexpanded!()
    }
}

impl<T: Numeric> StagingExpand<Tile<T>> {
    pub fn __expand_fill_method<F>(&mut self, scope: &Scope, fill: F)
    where
        F: FnOnce(&Scope, &mut TileExpand<T>, &PipelineExpand),
    {
        self.__expand_acquire_write_method(scope);
        fill(scope, &mut self.data, &self.pipeline);
        self.__expand_release_write_method(scope);
    }

    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<T>),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data);
        self.__expand_release_read_method(scope);
    }
}

/// Allocate one staged operand for `residence`. A gathered operand keeps its compacted physical
/// window and projection, so staging does not replicate each logical element for every gather
/// tap. The leaf performs the gather on read instead; that keeps staging compact but means
/// adjacent logical regions may re-read overlapping halo cells.
#[cube]
fn stage_operand<T: Numeric>(
    input: &Tile<T>,
    #[comptime] out: Space,
    #[comptime] residence: Residence,
) -> Tile<T> {
    let gathered = input.gathered();
    match comptime!(residence) {
        // Nothing is allocated and nothing is stored: the payload *is* the operand. It keeps the
        // source's own space, undivided, so the read has a grid to select this region's window
        // (or block of fragments) out of, which is [`read_operand`]'s job rather than a fill's.
        Residence::InPlace => match &input.tile_kind {
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => input.clone(),
            // A tensor map is not element-addressable: there is no window to select down to, and
            // its only sink is a hardware bulk copy into shared memory.
            TileKind::TmaGmem(_) => {
                panic!("Staging: a TMA source cannot be read in place; give it Residence::Smem")
            }
        },
        Residence::Register => {
            let delivery = input.delivery();
            comptime!(assert!(
                !delivery.is_tma(),
                "Staging: a TMA source cannot stage into plane tiles"
            ));
            comptime!(assert!(
                !gathered,
                "Staging: a gathered operand cannot stage into plane tiles (Residence::Register); \
                 only Residence::Smem stages one, as the compacted window its reader takes"
            ));
            let kind = comptime!(input.space.instruction().expect(
                "Staging: an operand staging into registers needs the space to state which \
                 instruction they are for; add `.instruction(...)` to its tiling"
            ));
            PlanePartition::store(
                comptime!(input.space.divide()),
                comptime!(kind),
                comptime!(out.clone()),
            )
        }
        Residence::Smem => MemData::smem_like(input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A one-level space over `M`/`N`/`K`, plus a projection per operand so a slot can be planned
    /// against it. `lhs` spans `M`/`K`, `rhs` spans `K`/`N`, so a `K` walk moves both.
    fn spaces() -> (Space, Space, Space) {
        let space = Tiling::new()
            .extents(&[(M, 8), (N, 8), (K, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, Cut::sequential(4))
                    .axis(N, Cut::sequential(4))
                    .axis(K, Cut::sequential(4))
            })
            .build();
        let lhs = space.project(&[M, K]);
        let rhs = space.project(&[K, N]);
        (space, lhs, rhs)
    }

    fn operand(residence: Residence, delivery: Delivery, space: &Space) -> SlotOperand<'_> {
        SlotOperand::new(residence, StageSource::Transport(delivery), space)
    }

    /// An operand a level above already materialized into registers: no delivery, since nothing
    /// moves it.
    fn fragment(residence: Residence, space: &Space) -> SlotOperand<'_> {
        SlotOperand::new(residence, StageSource::ResidentFragment, space)
    }

    #[test]
    fn an_in_place_operand_does_not_join_the_slot_rendezvous() {
        let (_, lhs, rhs) = spaces();
        let deliveries = rendezvous_deliveries(&[
            operand(Residence::InPlace, Delivery::Procedural, &lhs),
            operand(Residence::Smem, Delivery::Copy, &rhs),
        ]);
        assert_eq!(deliveries, vec![Delivery::Copy]);
        assert_eq!(slot_sync(&deliveries).0, Sync::Cube);
    }

    #[test]
    fn a_plane_stage_cannot_share_a_slot_with_smem() {
        assert!(!compatible_slot_residences(&[
            Residence::Register,
            Residence::Smem,
        ]));
        assert!(!compatible_slot_residences(&[
            Residence::Smem,
            Residence::Register,
        ]));
        assert!(compatible_slot_residences(&[
            Residence::Register,
            Residence::InPlace,
        ]));
    }

    #[test]
    #[should_panic(
        expected = "a procedural recipe stages in Smem and cannot share a slot with a Plane operand"
    )]
    fn a_procedural_recipe_cannot_share_a_slot_with_a_plane_stage() {
        let (space, lhs, rhs) = spaces();
        SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Procedural, &lhs),
                operand(Residence::Register, Delivery::Copy, &rhs),
            ],
            &space,
        );
    }

    #[test]
    #[should_panic(expected = "Plane and Smem operands cannot share a slot")]
    fn plane_and_smem_operands_cannot_share_a_slot() {
        let (space, lhs, rhs) = spaces();
        SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Copy, &lhs),
                operand(Residence::Register, Delivery::Copy, &rhs),
            ],
            &space,
        );
    }

    /// A slot holds a resident fragment where it lies, beside an operand that materializes: it is
    /// only re-materializing one that has no source to read.
    #[test]
    fn a_resident_fragment_is_held_only_where_it_lies() {
        let (_, lhs, rhs) = spaces();
        assert!(slot_admits_operands(&[
            fragment(Residence::InPlace, &lhs),
            operand(Residence::Smem, Delivery::Copy, &rhs),
        ]));
        assert!(!slot_admits_operands(&[fragment(Residence::Smem, &lhs)]));
        assert!(!slot_admits_operands(&[fragment(
            Residence::Register,
            &lhs
        )]));
    }

    /// Registers hold it already, so nothing fills it: every slot of the ring names the same
    /// fragments and each region selects its own block at the read.
    #[test]
    fn a_resident_fragment_is_never_filled() {
        let (space, lhs, rhs) = spaces();
        let plan = SlotPlan::new(
            &[
                fragment(Residence::InPlace, &lhs),
                operand(Residence::Smem, Delivery::Copy, &rhs),
            ],
            &space,
        );
        assert_eq!(
            plan.operand_plan(LHS, FIRST_SLOT).payload,
            SlotPayload::AtRegion
        );
        assert_eq!(plan.operand_plan(LHS, 1).payload, SlotPayload::AtRegion);
        assert!(!plan.reuses_first_buffer(LHS, 1));
        // It joins no rendezvous either: only the staged operand's delivery sets the sync.
        assert_eq!(plan.sync(), Sync::Cube);
    }

    /// An in-place operand allocates nothing, so no slot can fill it and no walk can fix it: the
    /// read goes to the source for every region, which is what the ring costs such a level.
    #[test]
    fn an_in_place_operand_is_read_at_its_region_however_the_walk_moves() {
        let (space, lhs, rhs) = spaces();
        // `rhs` spans no walked axis here, so a materialized operand would come out `Fixed`.
        let invariant = space.project(&[K]);
        for op_space in [&space, &invariant] {
            let plan = SlotPlan::new(
                &[
                    operand(Residence::InPlace, Delivery::Copy, &lhs),
                    operand(Residence::InPlace, Delivery::Copy, &rhs),
                ],
                op_space,
            );
            for slot in 0..2 {
                assert_eq!(plan.operand_plan(LHS, slot).payload, SlotPayload::AtRegion);
                assert_eq!(plan.operand_plan(RHS, slot).payload, SlotPayload::AtRegion);
                assert!(!plan.reuses_first_buffer(RHS, slot));
            }
        }
    }

    /// The two ends fragment selection arrives from: a slot that staged the operand into plane
    /// tiles, and one holding a grid a level above left resident. Either forces the unrolled walk
    /// ([`stage_walk_unrolled`]), so a resident fragment cannot be left to the output's own kind to
    /// answer for.
    #[test]
    fn either_end_of_a_fragment_read_asks_for_the_unrolled_walk() {
        let (space, lhs, rhs) = spaces();
        let resident = SlotPlan::new(&[fragment(Residence::InPlace, &lhs)], &space);
        assert!(resident.operand_plan(LHS, FIRST_SLOT).reads_fragments());

        let staged = SlotPlan::new(
            &[operand(Residence::Register, Delivery::Copy, &lhs)],
            &space,
        );
        assert!(staged.operand_plan(LHS, FIRST_SLOT).reads_fragments());

        // A window is a window wherever it lives: nothing here is selected by coordinate.
        for windowed in [
            SlotPlan::new(&[operand(Residence::Smem, Delivery::Copy, &rhs)], &space),
            SlotPlan::new(&[operand(Residence::InPlace, Delivery::Copy, &rhs)], &space),
        ] {
            assert!(!windowed.operand_plan(LHS, FIRST_SLOT).reads_fragments());
        }
    }

    /// The headline mix: one operand takes a shared stage while the other is read where it lies.
    /// The slot then rendezvouses for the staged one alone, which is what lets the pair share it.
    #[test]
    fn a_staged_operand_shares_a_slot_with_one_read_in_place() {
        let (_, lhs, rhs) = spaces();
        let deliveries = rendezvous_deliveries(&[
            operand(Residence::Smem, Delivery::Copy, &lhs),
            operand(Residence::InPlace, Delivery::Copy, &rhs),
        ]);
        assert_eq!(deliveries, vec![Delivery::Copy]);
    }

    /// Nothing materializes, so there is nothing to synchronize: the slot goes solo, which is what
    /// costs an all-in-place level nothing beyond the reads it would have made anyway.
    #[test]
    fn an_all_in_place_slot_synchronizes_nothing() {
        let (_, lhs, rhs) = spaces();
        let deliveries = rendezvous_deliveries(&[
            operand(Residence::InPlace, Delivery::Copy, &lhs),
            operand(Residence::InPlace, Delivery::Procedural, &rhs),
        ]);
        assert!(deliveries.is_empty());
        assert_eq!(slot_sync(&deliveries).0, Sync::Solo);
    }

    /// The walk moves both operands' windows here (every axis is cut), so nothing is fixed and
    /// every slot of the ring needs its own buffer.
    #[test]
    fn a_streamed_operand_is_rebuilt_in_every_slot() {
        let (space, lhs, rhs) = spaces();
        let plan = SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Copy, &lhs),
                operand(Residence::Smem, Delivery::Copy, &rhs),
            ],
            &space,
        );
        for slot in 0..3 {
            assert_eq!(
                plan.operand_plan(LHS, slot).payload,
                SlotPayload::Windowed(WindowMode::Streamed)
            );
            assert_eq!(
                plan.operand_plan(RHS, slot).payload,
                SlotPayload::Windowed(WindowMode::Streamed)
            );
            assert!(!plan.reuses_first_buffer(LHS, slot));
            assert!(!plan.reuses_first_buffer(RHS, slot));
        }
    }

    /// A `K`-only walk leaves the `M`/`N` output operand's window fixed. It is filled once, so one
    /// buffer serves the whole ring however deep: the first slot owns it and the rest reuse it.
    #[test]
    fn a_fixed_smem_operand_reuses_its_buffer_for_the_rest_of_the_ring() {
        let (space, lhs, _) = spaces();
        let invariant = space.project(&[M, N]);
        let plan = SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Copy, &lhs),
                operand(Residence::Smem, Delivery::Copy, &invariant),
            ],
            &space.project(&[K]),
        );
        assert_eq!(
            plan.operand_plan(LHS, FIRST_SLOT).payload,
            SlotPayload::Windowed(WindowMode::Streamed)
        );
        assert_eq!(
            plan.operand_plan(RHS, FIRST_SLOT).payload,
            SlotPayload::Windowed(WindowMode::Fixed)
        );
        assert_eq!(
            plan.operand_plan(RHS, 1).payload,
            SlotPayload::Windowed(WindowMode::Reused)
        );
        assert_eq!(
            plan.operand_plan(RHS, 2).payload,
            SlotPayload::Windowed(WindowMode::Reused)
        );
        assert!(plan.reuses_first_buffer(RHS, 1));
        assert!(!plan.reuses_first_buffer(RHS, FIRST_SLOT));
    }

    /// Two operands the walk leaves equally invariant, one shared and one in place. Fixing is a
    /// saving on a *fill*, so only the shared one takes it: the in-place one has no buffer to fill
    /// once and none for a later slot to reuse, so it reads at its region like any other.
    #[test]
    fn only_a_shared_operand_is_fixed_beside_an_in_place_one() {
        let (space, _, _) = spaces();
        let invariant = space.project(&[M, N]);
        let plan = SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Copy, &invariant),
                operand(Residence::InPlace, Delivery::Copy, &invariant),
            ],
            &space.project(&[K]),
        );
        assert_eq!(
            plan.operand_plan(LHS, FIRST_SLOT).payload,
            SlotPayload::Windowed(WindowMode::Fixed)
        );
        assert_eq!(
            plan.operand_plan(LHS, 1).payload,
            SlotPayload::Windowed(WindowMode::Reused)
        );
        assert!(plan.reuses_first_buffer(LHS, 1));

        assert_eq!(
            plan.operand_plan(RHS, FIRST_SLOT).payload,
            SlotPayload::AtRegion
        );
        assert_eq!(plan.operand_plan(RHS, 1).payload, SlotPayload::AtRegion);
        assert!(!plan.reuses_first_buffer(RHS, 1));
    }

    /// A TMA pair keeps the joint per-region fill, so no operand is fixed and no buffer is reused:
    /// its barrier arrives `full` once per fill, and splitting an invariant out would corrupt the
    /// phase.
    #[test]
    fn a_tma_slot_keeps_all_operands_streamed() {
        let (space, lhs, _) = spaces();
        let invariant = space.project(&[M, N]);
        let plan = SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Tma, &lhs),
                operand(Residence::Smem, Delivery::Tma, &invariant),
            ],
            &space.project(&[K]),
        );
        assert_eq!(
            plan.operand_plan(RHS, FIRST_SLOT).payload,
            SlotPayload::Windowed(WindowMode::Streamed)
        );
        assert_eq!(
            plan.operand_plan(RHS, 1).payload,
            SlotPayload::Windowed(WindowMode::Streamed)
        );
        assert_eq!(plan.sync(), Sync::Barrier);
    }
}
