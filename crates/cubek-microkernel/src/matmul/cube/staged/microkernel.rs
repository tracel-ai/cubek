use super::base::CubeStagedDrain;
use crate::matmul::cube::CubeStagedStore;
use crate::matmul::plane::{PlaneFragmentRowsAlong, plane_fragment_microkernel};
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::launch::{
    OpenedDestination, OpenedDestinationExpand, ServedDestination, ServedDestinationExpand,
};
use cubecl::tile::stage::{AsyncStage, AsyncStageExpand, OperandPair, Role};
use cubecl::tile::{Instruction, Region, Scratch, Semiring, StageStorage, Stages, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_std::microkernel::StageRing;

/// The comptime decisions a cube's walk reads, kept apart from the microkernel's layout and output
/// so the walk runs from these alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedWalk {
    pub drain: CubeStagedDrain,
    pub filling_planes: usize,
    pub ring: StageRing,
    pub scratch: Scratch,
    pub store: CubeStagedStore,
    /// How the planes read each operand: as bound, or dequantized at the element they contract,
    /// which a stage the fill decodes holds too.
    pub reads: MatmulOperands<MatmulRead>,
    /// The dimension its planes' instruction lays its rows along, which the sum is opened for.
    pub rows_along: PlaneFragmentRowsAlong,
}

impl CubeStagedWalk {
    /// The element the fill decodes its packed inputs into, which their stages hold: `None` where
    /// it stages every input as it lies.
    pub fn decoded(&self) -> Option<ElemType> {
        self.reads.lhs.staged().or(self.reads.rhs.staged())
    }
}

#[cube]
impl CubeStagedWalk {
    /// A zeroed sum over `over`'s box, its planes reading `a` and `b` as `walk` says: opened on
    /// the factor the instruction's rows lie along first, which fills its `A`, in the instruction
    /// the tile layer picks for them.
    pub fn open_sum<EA: Numeric>(
        over: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] walk: CubeStagedWalk,
    ) -> Tile {
        let mut sum = match walk.rows_along {
            PlaneFragmentRowsAlong::M => over.accumulator::<EA>(a, b, Semiring::SumProd),
            PlaneFragmentRowsAlong::N => over.accumulator::<EA>(b, a, Semiring::SumProd),
        };

        sum.zero();
        sum
    }

    /// `input` as the walk's stages hold it: as it lies, its scales kept beside its values, where
    /// the fill copies the inputs as stored and the planes decode them as they load; as it is
    /// where the fill decodes it. A float input carries no scales and stages as before.
    pub(crate) fn as_staged(input: &Tile, #[comptime] walk: CubeStagedWalk) -> Tile {
        if walk.decoded().is_none() {
            input.scales_kept_beside()
        } else {
            input.clone()
        }
    }

    /// `sum`, walked, bouncing its fragments through the scratch `walk` holds as it drains. The
    /// scratch is opened once the walk has freed its stages, past a cube barrier, so it takes
    /// their bytes rather than its own.
    pub fn drain_sum(sum: Tile, #[comptime] walk: CubeStagedWalk) -> Tile {
        if walk.scratch != Scratch::None {
            sync_cube();
        }

        sum.with_scratch(walk.scratch)
    }

    /// The sum of `a · b` over `run`, opened over `over`, returned rather than drained: every
    /// plane fills the stages and contracts them, or planes set aside fill them while the rest
    /// contract.
    fn sum_planes<EA: Numeric>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        over: &Tile,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) -> PlanesSum {
        if walk.filling_planes == 0 {
            Self::every_plane_fills::<EA>(run, a, b, over, walk, storage)
        } else {
            Self::planes_set_aside::<EA>(run, a, b, over, walk, storage)
        }
    }

    /// The sum of `a · b` over `run`, opened over `over`, every plane filling the stages and
    /// contracting them ([`cube_staged_sum_microkernel`]).
    fn every_plane_fills<EA: Numeric>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        over: &Tile,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) -> PlanesSum {
        let (a, b) = (Self::as_staged(a, walk), Self::as_staged(b, walk));
        let sum = Self::open_sum::<EA>(over, &a, &b, walk);
        let (a, a_factor) = a.factor(&sum);
        let (b, b_factor) = b.factor(&sum);

        cube_staged_sum_microkernel(run, &sum, &a, &b, walk, storage);

        PlanesSum {
            sum: Self::drain_sum(sum, walk)
                .scaled(&a_factor)
                .scaled(&b_factor),
            role: Role::new_Compute(),
        }
    }

    /// The sum of `a · b` over `run`, opened over `over`, planes set aside to fill the stages:
    /// those walk the slots filling each a lap ahead and do nothing else, and the rest walk them
    /// contracting. The two meet only on each slot's barriers, which is why no fragment bounces
    /// there: a scratch's cube barriers are ones the filling planes never reach. The ring's own
    /// walk has every plane fill, so the contracting planes walk the slots themselves; a plane
    /// group MMA's stage is freed only once the reads it issued complete
    /// ([`Stages::consume_async`]).
    ///
    /// Under the TMA store the ring is freed and the whole cube meets on a cube barrier before
    /// the sum returns, so a box declared next takes the stages' bytes with no plane still
    /// reading them. Every plane opens the sum, so it outlives the walk; a filling plane never
    /// touches its own.
    fn planes_set_aside<EA: Numeric>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        over: &Tile,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) -> PlanesSum {
        let (a, b) = (Self::as_staged(a, walk), Self::as_staged(b, walk));
        let sum = Self::open_sum::<EA>(over, &a, &b, walk);
        let (a, a_factor) = a.factor(&sum);
        let (b, b_factor) = b.factor(&sum);

        let depth = walk.ring.stages_in_flight.get();
        let mut stages = Stages::smem(&run, &a, &b, storage, depth, walk.decoded());

        // Both sets drive the same ring, so both count its laps alike: `depth` stages a lap, the
        // last lap short wherever the walk ends.
        let total = run.total();
        let laps = total.div_ceil(depth);

        match stages.role() {
            Role::Fill => {
                for lap in 0..laps {
                    #[unroll]
                    for slot in 0..depth {
                        let index = lap * depth + slot;
                        if index < total {
                            stages.fill(slot, &run.region(index));
                        }
                    }
                }
            }
            Role::Compute => {
                let instruction = sum.sum_instruction();
                if comptime!(instruction == Instruction::Wgmma) {
                    let issue = PlaneGroupStages {
                        sum: sum.clone(),
                        walk,
                    };

                    stages.consume_async::<PlaneGroupStages>(&run, &issue);
                } else {
                    for lap in 0..laps {
                        #[unroll]
                        for slot in 0..depth {
                            let index = lap * depth + slot;
                            if index < total {
                                let stage = run.region(index);
                                let sum_stage = sum.at(&stage);
                                stages.consume(slot, |a_stage, b_stage| {
                                    Self::contract_stage(
                                        &stage, &sum_stage, a_stage, b_stage, walk,
                                    );
                                });
                            }
                        }
                    }
                }
            }
        }

        Self::free_ring_for_box(&stages, walk);

        PlanesSum {
            sum: sum.scaled(&a_factor).scaled(&b_factor),
            role: stages.role(),
        }
    }

    /// Under the TMA store, the ring freed and the cube met on a barrier, so the box declared next
    /// takes the stages' bytes with no plane still reading them.
    fn free_ring_for_box(stages: &Stages<OperandPair>, #[comptime] walk: CubeStagedWalk) {
        // One branch holding two statements: split out, the pair would name nothing of its own.
        if comptime!(walk.store == CubeStagedStore::Tma) {
            stages.free();
            sync_cube();
        }
    }

    /// `c = a · b` over `run`, drained into a box in shared memory laid as `c` and handed to
    /// `dest` at `cube`, which a storing destination over a tensor map bulk-stores: the box
    /// declared after the walk where planes are set aside to fill, so it takes the ring's bytes,
    /// before it otherwise.
    fn store_by_tma<EA: Numeric, D: ServedDestination>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        dest: &D,
        cube: &Region,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) {
        if walk.filling_planes > 0 {
            Self::stage_after_walk::<EA, D>(run, a, b, dest, cube, walk, storage)
        } else {
            Self::stage_before_walk::<EA, D>(run, a, b, dest, cube, walk, storage)
        }
    }

    fn stage_after_walk<EA: Numeric, D: ServedDestination>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        dest: &D,
        cube: &Region,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) {
        let c = dest.tile().at(cube);
        let sum = Self::sum_planes::<EA>(run, a, b, &c, walk, storage);
        let staged = c.smem_box();
        sum.drained_into(&staged);
        staged.consumed_by::<D>(dest, cube);
    }

    fn stage_before_walk<EA: Numeric, D: ServedDestination>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        dest: &D,
        cube: &Region,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) {
        let staged = dest.tile().at(cube).smem_box();
        Self::sum_planes::<EA>(run, a, b, &staged, walk, storage).drained_into(&staged);
        staged.consumed_by::<D>(dest, cube);
    }

    /// One stage contracted into `sum_stage`, the sum's window of it: every plane runs the plane
    /// microkernel on its window of `stage`, the cube's region of the stage, out of `a_stage` and
    /// `b_stage`, the stage's operands.
    fn contract_stage(
        stage: &Region,
        sum_stage: &Tile,
        a_stage: &Tile,
        b_stage: &Tile,
        #[comptime] walk: CubeStagedWalk,
    ) {
        for plane in stage {
            plane_fragment_microkernel(
                &plane,
                &sum_stage.at(&plane),
                &a_stage.at(&plane),
                &b_stage.at(&plane),
                None,
                walk.reads,
            );
        }
    }
}

/// One cube's box of the output, `c = a · b` over `run`, its run of `K`: the cube stages both
/// operands a step of `K` at a time and every plane contracts its tile out of each stage with the
/// plane microkernel. `storage` is how the stages hold their cells. The operands' scales ride `a`
/// and `b`, the stages keeping words of block scales beside the values; the ones above such words,
/// which the instruction cannot take, leave for the sum and are applied once, as it is written
/// ([`Tile::factor`]).
///
/// The cube's result is handed to `dest` at `cube` wherever it ends: each plane's fragments
/// ([`CubeStagedStore::Fragments`]); a box in shared memory laid as the output's, which the
/// planes drain into and the whole cube, filling planes too, hands over, a storing destination
/// bulk-storing it ([`CubeStagedStore::Tma`]), the box taking the ring's bytes beside planes set
/// aside to fill; or, splitting `K`, one accumulator in shared memory each plane adds its partial
/// into, atomically, handed over once the cube has synchronized ([`CubeStagedDrain::Merged`]).
/// Planes merging along `K` are never stored by the TMA engine.
#[cube]
pub fn cube_staged_microkernel<EA: Numeric, D: ServedDestination>(
    run: Walk,
    a: &Tile,
    b: &Tile,
    dest: &D,
    cube: &Region,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
) {
    let c = dest.tile().at(cube);

    match (walk.drain, walk.store) {
        (CubeStagedDrain::EachPlane, CubeStagedStore::Fragments) => {
            CubeStagedWalk::sum_planes::<EA>(run, a, b, &c, walk, storage)
                .consumed_by::<D>(dest, cube);
        }
        (CubeStagedDrain::EachPlane, CubeStagedStore::Tma) => {
            CubeStagedWalk::store_by_tma::<EA, D>(run, a, b, dest, cube, walk, storage);
        }
        (CubeStagedDrain::Merged, _) => {
            let accumulator = c.smem_accumulation::<EA>();
            CubeStagedWalk::sum_planes::<EA>(run, a, b, &accumulator.sink, walk, storage)
                .drained_into(&accumulator.sink);

            sync_cube();

            accumulator.source.consumed_by::<D>(dest, cube);
        }
    }
}

/// `sum += a · b` over `run`, a run of `K` of the box `over`, whose sum is already open: every
/// plane fills the stages a step of `K` at a time and contracts its tile out of each with the plane
/// microkernel, and nothing drains.
#[cube]
pub fn cube_staged_sum_microkernel(
    run: Walk,
    sum: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
) {
    let in_flight = walk.ring.stages_in_flight.get();
    let mut stages = Stages::smem(&run, a, b, storage, in_flight, walk.decoded());

    stages.walk(run, walk.ring.prefetch, |slot, stage| {
        let sum_stage = sum.at(stage);
        slot.consume(|a_stage, b_stage| {
            CubeStagedWalk::contract_stage(stage, &sum_stage, a_stage, b_stage, walk);
        });
    });

    stages.free();
}

/// What a cube's compute planes issue on each stage of a plane group MMA's walk: every plane
/// group's MMAs of the stage into `sum`, whose completion frees the stage once they are done with
/// it ([`Stages::consume_async`]).
#[derive(CubeType)]
struct PlaneGroupStages {
    sum: Tile,
    #[cube(comptime)]
    walk: CubeStagedWalk,
}

#[cube]
impl AsyncStage for PlaneGroupStages {
    fn issue(&self, region: &Region, lhs: &Tile, rhs: &Tile) -> Pending<()> {
        let sum_stage = self.sum.at(region);
        CubeStagedWalk::contract_stage(region, &sum_stage, lhs, rhs, self.walk);
        self.sum.commit()
    }

    fn done(&self) -> Pending<()> {
        self.sum.commit()
    }
}

/// A cube's finished sum, its scales riding it, and whether this plane holds a share of it.
#[derive(CubeType)]
struct PlanesSum {
    sum: Tile,
    role: Role,
}

impl core::fmt::Debug for PlanesSum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PlanesSum").finish_non_exhaustive()
    }
}

#[cube]
impl PlanesSum {
    fn drained_into(self, into: &Tile) {
        match self.role {
            Role::Compute => self.sum.drained_into(into),
            Role::Fill => {}
        }
    }

    fn consumed_by<D: ServedDestination>(self, dest: &D, cube: &Region) {
        let mut opened = dest.open(cube, &self.sum);
        match self.role {
            Role::Compute => opened.consume(cube, &self.sum),
            Role::Fill => {}
        }
        opened.close();
    }
}
