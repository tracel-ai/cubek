use super::base::CubeStagedDrain;
use crate::microkernel::dense::cube::CubeStagedStore;
use crate::microkernel::dense::plane::{
    PlaneFragmentStepSum, plane_fragment_microkernel, plane_fragment_partial,
};
use crate::microkernel::staging::StageRing;
use crate::microkernel::{MatmulOperands, MatmulRead};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_tile::{
    Instruction, Region, Scratch, Semiring, StageStorage, Stages, Tile, Walk,
    stage::{AsyncStage, AsyncStageExpand, Role},
};

/// The comptime decisions a cube's walk reads, kept apart from the microkernel's layout and output
/// so the walk runs from these alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedWalk {
    pub drain: CubeStagedDrain,
    pub filling_planes: usize,
    pub ring: StageRing,
    pub scratch: Scratch,
    pub instruction: Instruction,
    pub step_sum: PlaneFragmentStepSum,
    pub store: CubeStagedStore,
    /// How the planes read each operand: as bound, or dequantized at the element they contract,
    /// which a stage the fill decodes holds too.
    pub reads: MatmulOperands<MatmulRead>,
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
    /// A zeroed sum over `over`'s box, its planes reading `a` and `b` as `walk` says.
    pub fn open_sum<EA: Numeric>(
        over: &Tile,
        a: &Tile,
        b: &Tile,
        #[comptime] walk: CubeStagedWalk,
    ) -> Tile {
        let mut sum = over.accumulator::<EA>(a, b, comptime!(walk.instruction), Semiring::SUM_PROD);
        sum.zero();
        sum
    }

    /// `sum`, walked, bouncing its fragments through the scratch `walk` holds as it drains. The
    /// scratch is opened once the walk has freed its stages, past a cube barrier, so it takes
    /// their bytes rather than its own.
    pub fn drain_sum(sum: Tile, #[comptime] walk: CubeStagedWalk) -> Tile {
        if comptime!(walk.scratch != Scratch::None) {
            sync_cube();
        }
        sum.with_scratch(comptime!(walk.scratch))
    }

    /// `c = a · b` over `run`, each plane draining its own tile of `c`: every plane fills the
    /// stages and contracts them, or planes set aside fill them while the rest contract.
    fn drain_planes<EA: Numeric, EL: Numeric>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        c: &Tile,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) {
        if comptime!(walk.filling_planes == 0) {
            Self::every_plane_fills::<EA, EL>(run, a, b, c, walk, storage);
        } else {
            Self::planes_set_aside::<EA, EL>(run, a, b, c, walk, storage);
        }
    }

    /// `c = a · b` over `run`, every plane filling the stages and contracting them
    /// ([`cube_staged_sum_microkernel`]).
    fn every_plane_fills<EA: Numeric, EL: Numeric>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        c: &Tile,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) {
        let mut c = c.clone();
        let (a, a_factor) = a.factor(c.space(), comptime!(walk.instruction));
        let (b, b_factor) = b.factor(c.space(), comptime!(walk.instruction));
        let sum = Self::open_sum::<EA>(&c, &a, &b, walk);
        cube_staged_sum_microkernel::<EL>(run, &sum, &c, &a, &b, walk, storage);
        let sum = Self::drain_sum(sum, walk);
        c.dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));
    }

    /// `c = a · b` over `run`, planes set aside to fill the stages: those walk the slots filling
    /// each a lap ahead and do nothing else, and the rest walk them contracting, then drain. The
    /// two meet only on each slot's barriers, which is why no fragment bounces there: a scratch's
    /// cube barriers are ones the filling planes never reach. The ring's own walk has every plane
    /// fill, so the contracting planes walk the slots themselves; a warpgroup MMA's stage is freed
    /// only once the reads it issued complete ([`Stages::consume_async`]).
    fn planes_set_aside<EA: Numeric, EL: Numeric>(
        run: Walk,
        a: &Tile,
        b: &Tile,
        c: &Tile,
        #[comptime] walk: CubeStagedWalk,
        #[comptime] storage: StageStorage,
    ) {
        let mut c = c.clone();
        let (a, a_factor) = a.factor(c.space(), comptime!(walk.instruction));
        let (b, b_factor) = b.factor(c.space(), comptime!(walk.instruction));
        let depth = comptime!(walk.ring.stages_in_flight.get());
        let mut stages = Stages::smem(&run, &a, &b, storage, depth, comptime!(walk.decoded()));
        // Both sets drive the same ring, so both count its laps alike: `depth` stages a lap, the
        // last lap short wherever the walk ends.
        let total = run.total();
        let laps = (total + comptime!(depth - 1)) / comptime!(depth);
        match stages.role() {
            Role::Fill => {
                for lap in 0..laps {
                    #[unroll]
                    for slot in 0..depth {
                        let index = lap * comptime!(depth) + slot;
                        if index < total {
                            stages.fill(slot, &run.region(index));
                        }
                    }
                }
            }
            Role::Compute => {
                let sum = Self::open_sum::<EA>(&c, &a, &b, walk);
                let partial = plane_fragment_partial::<EL>(
                    &c,
                    &a,
                    &b,
                    comptime!(walk.step_sum),
                    comptime!(walk.instruction),
                );
                if comptime!(walk.instruction == Instruction::Wgmma) {
                    let issue = WarpgroupStages {
                        sum: sum.clone(),
                        partial: partial.clone(),
                        walk,
                    };
                    stages.consume_async::<WarpgroupStages>(&run, &issue);
                } else {
                    for lap in 0..laps {
                        #[unroll]
                        for slot in 0..depth {
                            let index = lap * comptime!(depth) + slot;
                            if index < total {
                                let stage = run.region(index);
                                let sum_stage = sum.at(&stage);
                                stages.consume(slot, |a_stage, b_stage| {
                                    Self::contract_stage(
                                        &stage, &sum_stage, &partial, a_stage, b_stage, walk,
                                    );
                                });
                            }
                        }
                    }
                }
                c.dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));
            }
        }
    }

    /// One stage contracted into `sum_stage`, the sum's window of it: every plane runs the plane
    /// microkernel on its window of `stage`, the cube's region of the stage, out of `a_stage` and
    /// `b_stage`, the stage's operands.
    fn contract_stage(
        stage: &Region,
        sum_stage: &Tile,
        partial: &ComptimeOption<Tile>,
        a_stage: &Tile,
        b_stage: &Tile,
        #[comptime] walk: CubeStagedWalk,
    ) {
        for plane in stage {
            plane_fragment_microkernel(
                &plane,
                &sum_stage.at(&plane),
                partial,
                &a_stage.at(&plane),
                &b_stage.at(&plane),
                comptime!(walk.instruction),
                None,
                comptime!(walk.reads),
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
/// The planes drain where `walk` says: each its own tile of `c`; each into a box in shared memory
/// laid as `c`'s, which the whole cube, filling planes too, then stores in one bulk copy
/// ([`CubeStagedStore::Tma`]); or, splitting `K`, each adding its partial into one accumulator in
/// shared memory, atomically, which the cube copies out once it has synchronized
/// ([`CubeStagedDrain::Merged`]). Planes merging along `K` are never stored by the TMA engine.
#[cube]
pub fn cube_staged_microkernel<EA: Numeric, EL: Numeric>(
    run: Walk,
    a: &Tile,
    b: &Tile,
    c: &Tile,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
) {
    let mut c = c.clone();
    match comptime!((walk.drain, walk.store)) {
        (CubeStagedDrain::EachPlane, CubeStagedStore::Fragments) => {
            CubeStagedWalk::drain_planes::<EA, EL>(run, a, b, &c, walk, storage);
        }
        (CubeStagedDrain::EachPlane, CubeStagedStore::Tma) => {
            let staged = c.smem_box();
            CubeStagedWalk::drain_planes::<EA, EL>(run, a, b, &staged, walk, storage);
            c.copy_from(&staged);
        }
        (CubeStagedDrain::Merged, _) => {
            let accumulator = c.smem_accumulation::<EA>();
            CubeStagedWalk::drain_planes::<EA, EL>(run, a, b, &accumulator.sink, walk, storage);
            sync_cube();
            c.copy_from(&accumulator.source);
        }
    }
}

/// `sum += a · b` over `run`, a run of `K` of the box `over`, whose sum is already open: every
/// plane fills the stages a step of `K` at a time and contracts its tile out of each with the
/// plane microkernel, and nothing drains. A run that arrives as boxes adds each into one sum, as a
/// hypercube's cube walks its run
/// ([`hypercube_staged_relay_microkernel`](crate::microkernel::dense::hypercube::hypercube_staged_relay_microkernel())).
#[cube]
pub fn cube_staged_sum_microkernel<EL: Numeric>(
    run: Walk,
    sum: &Tile,
    over: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] walk: CubeStagedWalk,
    #[comptime] storage: StageStorage,
) {
    let partial = plane_fragment_partial::<EL>(
        over,
        a,
        b,
        comptime!(walk.step_sum),
        comptime!(walk.instruction),
    );
    let mut stages = Stages::smem(
        &run,
        a,
        b,
        storage,
        comptime!(walk.ring.stages_in_flight.get()),
        comptime!(walk.decoded()),
    );
    stages.walk(run, comptime!(walk.ring.prefetch), |slot, stage| {
        let sum_stage = sum.at(stage);
        slot.consume(|a_stage, b_stage| {
            CubeStagedWalk::contract_stage(stage, &sum_stage, &partial, a_stage, b_stage, walk);
        });
    });
    stages.free();
}

/// What a cube's compute planes issue on each stage of a warpgroup MMA's walk: every plane
/// group's MMAs of the stage into `sum`, whose completion frees the stage once they are done with
/// it ([`Stages::consume_async`]).
#[derive(CubeType)]
struct WarpgroupStages {
    sum: Tile,
    partial: ComptimeOption<Tile>,
    #[cube(comptime)]
    walk: CubeStagedWalk,
}

#[cube]
impl AsyncStage for WarpgroupStages {
    fn issue(&self, region: &Region, lhs: &Tile, rhs: &Tile) -> Pending<()> {
        let sum_stage = self.sum.at(region);
        CubeStagedWalk::contract_stage(region, &sum_stage, &self.partial, lhs, rhs, self.walk);
        self.sum.commit()
    }

    fn done(&self) -> Pending<()> {
        self.sum.commit()
    }
}
