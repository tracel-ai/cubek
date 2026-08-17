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
    delivery: Delivery,
    /// Already materialized as a plane fragment a level above, so it has no payload to build.
    fragment: bool,
    space: &'a Space,
}

impl<'a> SlotOperand<'a> {
    pub(crate) fn new(
        residence: Residence,
        delivery: Delivery,
        fragment: bool,
        space: &'a Space,
    ) -> Self {
        SlotOperand {
            residence,
            delivery,
            fragment,
            space,
        }
    }
}

/// What every slot of one ring decides at comptime: how it rendezvouses, and per operand where it
/// lives and when it is filled. Deduced once from the operands, so a ring's slots agree by
/// construction rather than by each re-deriving the same answers off the same tiles.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct SlotPlan {
    residence: Vec<Residence>,
    fill_modes: Vec<FillMode>,
    sync: Sync,
    collective_full: bool,
}

impl SlotPlan {
    pub(crate) fn new(operands: &[SlotOperand], op_space: &Space) -> SlotPlan {
        // Admission first: an operand's delivery is only meaningful once it is known to be a fill
        // source at all, and a resident fragment never is.
        assert!(
            slot_admits_operands(operands),
            "Staging: an operand already materialized as a plane fragment cannot share a slot; \
             another operand asked to materialize at this level, so give it Residence::InPlace \
             here (materializing it at a level above instead)"
        );
        let residence: Vec<_> = operands.iter().map(|op| op.residence).collect();
        if !compatible_slot_residences(&residence) {
            if operands
                .iter()
                .any(|op| op.delivery == Delivery::Procedural)
            {
                panic!(
                    "Staging: a procedural recipe demotes to Smem and cannot share a slot with a Plane operand"
                );
            } else {
                panic!("Staging: Plane and Smem operands cannot share a slot");
            }
        }
        // Fix an operand only when its window is genuinely invariant across the walk. A barrier
        // pipeline arrives `full` once per fill, so a TMA pair keeps the joint per-region fill;
        // splitting an invariant out would corrupt its phase. A dynamic level can't decide
        // invariance at comptime. Both fall back to streaming.
        let can_fix_invariants =
            op_space.is_static() && !operands.iter().any(|op| op.delivery.is_tma());
        let fill_modes = operands
            .iter()
            .map(|op| {
                if can_fix_invariants && op_space.walk_invariant(op.space) {
                    FillMode::Fixed
                } else {
                    FillMode::Streamed
                }
            })
            .collect();
        let (sync, collective_full) = slot_sync(&rendezvous_deliveries(operands));
        SlotPlan {
            residence,
            fill_modes,
            sync,
            collective_full,
        }
    }

    pub(crate) fn residence(&self, operand: usize) -> Residence {
        self.residence[operand]
    }

    /// When operand `operand` is filled in ring slot `slot`. The first slot owns every payload;
    /// a later slot reuses the first slot's buffer for whatever the walk never rewrites.
    pub(crate) fn fill_mode(&self, operand: usize, slot: usize) -> FillMode {
        match slot {
            0 => self.fill_modes[operand],
            _ => self.fill_modes[operand].in_later_slot(self.residence[operand]),
        }
    }

    /// Whether slot `slot` takes the first slot's buffer for `operand` instead of allocating one.
    pub(crate) fn reuses_first_buffer(&self, operand: usize, slot: usize) -> bool {
        self.fill_mode(operand, slot) == FillMode::Reused
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

/// Only operands that physically materialize join the slot's rendezvous: an in-place payload is
/// rebound to the source region, moving no bytes and needing no barrier.
fn rendezvous_deliveries(operands: &[SlotOperand]) -> Vec<Delivery> {
    operands
        .iter()
        .filter(|op| op.residence == Residence::Smem)
        .map(|op| op.delivery)
        .collect()
}

/// Whether two operands can share one slot. Stated per pair rather than by counting kinds, so a
/// new [`Residence`] has to say how it meets each of the others instead of defaulting to
/// compatible.
fn compatible_residence_pair(a: Residence, b: Residence) -> bool {
    match (a, b) {
        // A plane partition is private to its unit and assumes a solo fill; a shared-memory stage
        // selects a slot-wide Cube or Barrier pipeline. One slot cannot rendezvous both ways.
        (Residence::Plane, Residence::Smem) | (Residence::Smem, Residence::Plane) => false,
        (Residence::Plane, Residence::Plane | Residence::InPlace) => true,
        (Residence::Smem, Residence::Smem | Residence::InPlace) => true,
        (Residence::InPlace, Residence::Plane | Residence::Smem | Residence::InPlace) => true,
        (Residence::Auto, _) | (_, Residence::Auto) => {
            panic!("Residence::Auto must be resolved before staging")
        }
    }
}

/// Whether every pair of `residences` can share one slot. The self-pair is included, which is what
/// catches an unresolved [`Auto`](Residence::Auto) on a slot holding a single operand.
fn compatible_slot_residences(residences: &[Residence]) -> bool {
    residences.iter().enumerate().all(|(i, &a)| {
        residences[i..]
            .iter()
            .all(|&b| compatible_residence_pair(a, b))
    })
}

/// A slot builds a payload for every operand it holds, and an operand a level above already
/// materialized into plane-private registers has none to build: it can be neither filled through
/// the pipeline nor rebound per region. Such a level has to leave every operand
/// [`InPlace`](Residence::InPlace), which lowers to the plain recursive walk and builds no slot at
/// all, so reaching here means one operand asked to materialize beside a fragment.
fn slot_admits_operands(operands: &[SlotOperand]) -> bool {
    !operands.iter().any(|op| op.fragment)
}

/// Fill a materialized operand through its slot pipeline, or rebind an in-place payload to the
/// source region. The pipeline is slot-wide while this decision is per-operand, which is what lets
/// one slot hold a staged operand beside one read where it lies.
#[cube]
pub(crate) fn fill_operand<T: Numeric>(
    dst: &mut Tile<T>,
    src: &Tile<T>,
    #[comptime] residence: Residence,
    pipe: &Pipeline,
) {
    if comptime!(residence == Residence::InPlace) {
        dst.rebind_from(src);
    } else {
        pipe.fill(dst, src);
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Ring<(Tile<Lhs>, Tile<Rhs>)> {
    /// Build the `depth` slots staging the operands `lhs`/`rhs`. An [`Residence::Plane`] operand
    /// stages into plane-private tile partitions ([`Solo`](Sync::Solo)); [`Smem`](Residence::Smem)
    /// takes fresh shared memory, with [`Sync`] deduced from the operands' delivery.
    ///
    /// Slot 0 allocates every payload. A later slot allocates only what the walk refills: an
    /// operand whose window the walk leaves fixed is filled once and never rewritten, so one
    /// buffer serves the whole ring ([`FillMode::Reused`]).
    pub fn binary(
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
        #[comptime] depth: usize,
    ) -> Ring<(Tile<Lhs>, Tile<Rhs>)> {
        let lhs_residence = lhs.residence(comptime!(&out));
        let rhs_residence = rhs.residence(comptime!(&out));
        let lhs_delivery = lhs.delivery();
        let rhs_delivery = rhs.delivery();
        let lhs_fragment = lhs.resident_fragment();
        let rhs_fragment = rhs.resident_fragment();
        let plan = comptime!(SlotPlan::new(
            &[
                SlotOperand::new(lhs_residence, lhs_delivery, lhs_fragment, &lhs.space),
                SlotOperand::new(rhs_residence, rhs_delivery, rhs_fragment, &rhs.space),
            ],
            &op_space,
        ));

        let mut slots = Sequence::<Staging<(Tile<Lhs>, Tile<Rhs>)>>::new();
        #[unroll]
        for slot in 0..depth {
            let staged_lhs = if comptime!(plan.reuses_first_buffer(0, slot)) {
                slots.index(0).data.0.clone()
            } else {
                stage_operand(lhs, comptime!(out.clone()), lhs_residence)
            };
            let staged_rhs = if comptime!(plan.reuses_first_buffer(1, slot)) {
                slots.index(0).data.1.clone()
            } else {
                stage_operand(rhs, comptime!(out.clone()), rhs_residence)
            };
            let staging = Staging::wrap(
                (staged_lhs, staged_rhs),
                Pipeline::new(comptime!(plan.sync()), comptime!(plan.collective_full())),
                comptime!(plan.fill_mode(0, slot)),
                comptime!(Option::Some(plan.fill_mode(1, slot))),
                lhs_residence,
                comptime!(Option::Some(rhs_residence)),
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
        let fixed_lhs = self.is_fixed();
        let fixed_rhs = self.is_fixed_rhs();
        let lhs_residence = self.residence();
        let rhs_residence = self.residence_rhs();
        if comptime!(fixed_lhs || fixed_rhs) {
            self.fill(|staged_operands, pipe| {
                if comptime!(fixed_lhs) {
                    let source = lhs.at(region);
                    fill_operand(&mut staged_operands.0, &source, lhs_residence, pipe);
                }
                if comptime!(fixed_rhs) {
                    let source = rhs.at(region);
                    fill_operand(&mut staged_operands.1, &source, rhs_residence, pipe);
                }
            });
        }
    }

    /// Fill the streamed operand(s), everything the walk moves, from `region`'s window. Runs per
    /// region inside the walk. The slot still rendezvouses when every operand is fixed: the
    /// pipeline's phase belongs to the slot, not to any one operand.
    pub fn fill_streamed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let stream_lhs = self.is_streamed();
        let stream_rhs = self.is_streamed_rhs();
        let lhs_residence = self.residence();
        let rhs_residence = self.residence_rhs();
        self.fill(|staged_operands, pipe| {
            if comptime!(stream_lhs) {
                let source = lhs.at(region);
                fill_operand(&mut staged_operands.0, &source, lhs_residence, pipe);
            }
            if comptime!(stream_rhs) {
                let source = rhs.at(region);
                fill_operand(&mut staged_operands.1, &source, rhs_residence, pipe);
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
    /// Each tile's bytes and runtime map were stored together by the producer.
    /// See [`StagingExpand::__expand_consume_method`].
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }

    /// Consumer for a fill no later fill will publish (the walk's final regions): publish
    /// the slot first, then consume. See [`StagingExpand::__expand_consume_final_method`].
    pub fn consume_final(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
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

    pub fn __expand_consume_final_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<Lhs>, &TileExpand<Rhs>),
    {
        self.__expand_publish_method(scope);
        self.__expand_consume_method(scope, compute);
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
        let delivery = input.delivery();
        let fragment = input.resident_fragment();
        let plan = comptime!(SlotPlan::new(
            &[SlotOperand::new(
                residence,
                delivery,
                fragment,
                &input.space
            )],
            &op_space,
        ));

        let mut slots = Sequence::<Staging<Tile<T>>>::new();
        #[unroll]
        for slot in 0..depth {
            let staged_input = if comptime!(plan.reuses_first_buffer(0, slot)) {
                slots.index(0).data.clone()
            } else {
                stage_operand(input, comptime!(out.clone()), residence)
            };
            let staging = Staging::wrap(
                staged_input,
                Pipeline::new(comptime!(plan.sync()), comptime!(plan.collective_full())),
                comptime!(plan.fill_mode(0, slot)),
                comptime!(Option::None),
                residence,
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
        let fixed = self.is_fixed();
        let residence = self.residence();
        if comptime!(fixed) {
            self.fill(|s, pipe| {
                let source = input.at(region);
                fill_operand(s, &source, residence, pipe);
            });
        }
    }

    /// Fill the operand from `region`'s window if the walk moves it. The slot still rendezvouses
    /// otherwise: the pipeline's phase belongs to the slot, not to the operand.
    pub fn fill_streamed(&mut self, input: &Tile<T>, region: &Region) {
        let stream = self.is_streamed();
        let residence = self.residence();
        self.fill(|s, pipe| {
            if comptime!(stream) {
                let source = input.at(region);
                fill_operand(s, &source, residence, pipe);
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
    /// Each tile's bytes and runtime map were stored together by the producer.
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<T>)) {
        unexpanded!()
    }

    /// Consumer for a fill no later fill will publish: publish the slot first, then consume.
    pub fn consume_final(&mut self, _compute: impl FnOnce(&Tile<T>)) {
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

    pub fn __expand_consume_final_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<T>),
    {
        self.__expand_publish_method(scope);
        self.__expand_consume_method(scope, compute);
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
    let delivery = input.delivery();
    match comptime!(residence) {
        // Nothing is allocated: the payload names the same bytes (or the same recipe) as the
        // source, one level down, and each region rebinds it rather than filling it.
        Residence::InPlace => match &input.tile_kind {
            TileKind::Procedural(data) => Tile::<T> {
                tile_kind: TileKind::new_Procedural(data.at_space(comptime!(input.space.divide()))),
                space: comptime!(input.space.divide()),
                leaf: comptime!(input.leaf),
            },
            TileKind::Gmem(data) => Tile::<T> {
                tile_kind: TileKind::new_Gmem(data.in_place_slot(comptime!(input.space.clone()))),
                space: comptime!(input.space.divide()),
                leaf: comptime!(input.leaf),
            },
            TileKind::Smem(data) => Tile::<T> {
                tile_kind: TileKind::new_Smem(data.in_place_slot(comptime!(input.space.clone()))),
                space: comptime!(input.space.divide()),
                leaf: comptime!(input.leaf),
            },
            // A tensor map is not element-addressable: there is no window to hand down, and its
            // only sink is a hardware bulk copy into shared memory.
            TileKind::TmaGmem(_) => {
                panic!("Staging: a TMA source cannot be read in place; give it Residence::Smem")
            }
            // Unreachable: `slot_admits_operands` turns a resident fragment away before any
            // payload is built, since it has no window to rebind.
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Staging: a resident fragment cannot be held by a slot")
            }
        },
        Residence::Plane => {
            comptime!(assert!(
                !delivery.is_tma(),
                "Staging: a TMA source cannot stage into plane tiles"
            ));
            comptime!(assert!(
                !gathered,
                "Staging: a gathered operand cannot stage into plane tiles (Residence::Plane); \
                 only Residence::Smem stages one, as the compacted window its leaf reads"
            ));
            PlanePartition::store(
                comptime!(input.space.divide()),
                comptime!(input.leaf),
                comptime!(out.clone()),
            )
        }
        Residence::Smem => MemData::smem_like(input),
        // `Tile::residence` resolves it against the level below before anything reaches here.
        Residence::Auto => panic!("Staging: Residence::Auto is a request, not a backing"),
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
        SlotOperand::new(residence, delivery, false, space)
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
            Residence::Plane,
            Residence::Smem,
        ]));
        assert!(!compatible_slot_residences(&[
            Residence::Smem,
            Residence::Plane,
        ]));
        assert!(compatible_slot_residences(&[
            Residence::Plane,
            Residence::InPlace,
        ]));
    }

    #[test]
    #[should_panic(
        expected = "a procedural recipe demotes to Smem and cannot share a slot with a Plane operand"
    )]
    fn a_procedural_recipe_cannot_share_a_slot_with_a_plane_stage() {
        let (space, lhs, rhs) = spaces();
        SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Procedural, &lhs),
                operand(Residence::Plane, Delivery::Copy, &rhs),
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
                operand(Residence::Plane, Delivery::Copy, &rhs),
            ],
            &space,
        );
    }

    /// An asymmetric plan can leave one operand a fragment while the other still asks to
    /// materialize. Both read `InPlace` at that point, so the residences alone look compatible;
    /// only the physical kind tells the pair apart.
    #[test]
    fn a_resident_fragment_is_turned_away_from_a_slot() {
        let (_, lhs, rhs) = spaces();
        let fragment = SlotOperand::new(Residence::InPlace, Delivery::Copy, true, &lhs);
        let staged = operand(Residence::Smem, Delivery::Copy, &rhs);
        assert!(!slot_admits_operands(&[fragment]));
        assert!(slot_admits_operands(&[staged]));
        // The residence check on its own waves the same pair through.
        assert!(compatible_slot_residences(&[
            Residence::InPlace,
            Residence::Smem,
        ]));
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
    /// makes an all-in-place level equivalent to the plain recursive walk.
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
            assert_eq!(plan.fill_mode(0, slot), FillMode::Streamed);
            assert_eq!(plan.fill_mode(1, slot), FillMode::Streamed);
            assert!(!plan.reuses_first_buffer(0, slot));
            assert!(!plan.reuses_first_buffer(1, slot));
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
        assert_eq!(plan.fill_mode(0, 0), FillMode::Streamed);
        assert_eq!(plan.fill_mode(1, 0), FillMode::Fixed);
        assert_eq!(plan.fill_mode(1, 1), FillMode::Reused);
        assert_eq!(plan.fill_mode(1, 2), FillMode::Reused);
        assert!(plan.reuses_first_buffer(1, 1));
        assert!(!plan.reuses_first_buffer(1, 0));
    }

    /// An in-place payload allocates nothing, so a later slot has no buffer to reuse: it builds
    /// its own view and rebinds it, or the rebind would only ever reach the first slot.
    #[test]
    fn a_fixed_in_place_operand_never_reuses_a_buffer() {
        let (space, lhs, _) = spaces();
        let invariant = space.project(&[M, N]);
        let plan = SlotPlan::new(
            &[
                operand(Residence::Smem, Delivery::Copy, &lhs),
                operand(Residence::InPlace, Delivery::Copy, &invariant),
            ],
            &space.project(&[K]),
        );
        assert_eq!(plan.fill_mode(1, 0), FillMode::Fixed);
        assert_eq!(plan.fill_mode(1, 1), FillMode::Fixed);
        assert!(!plan.reuses_first_buffer(1, 1));
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
        assert_eq!(plan.fill_mode(1, 0), FillMode::Streamed);
        assert_eq!(plan.fill_mode(1, 1), FillMode::Streamed);
        assert_eq!(plan.sync(), Sync::Barrier);
    }
}
