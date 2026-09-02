//! The output half of [`Residence`], where a residence is read as a *scope* rather than a refill.
//! Both directions share the vocabulary and differ in which way values travel: an input is
//! *filled* from its source into its residence, an output *drains* from its residence into its
//! sink.
//!
//! An output states its residence at the level whose region it lives across, the way an input
//! does. The output's own space has no contracted axis, so its outermost level already holds one
//! region per instance: stating [`Register`](Residence::Register) there opens a register-resident
//! accumulator for that whole region, and every level below runs inside it.
//!
//! ```ignore
//! let mut acc = c.accumulate::<EA, _>(&a, Monoid::Sum);  // opens the scope under its fold
//! acc.mm(&a, &b, Semiring::SUM_PROD);                    // `c = a·b`, and the drain that closes it
//! ```
//!
//! `mm` states `c = a·b`, so it owns the init. To fold onto what `c` already holds, state that
//! init and use the accumulating verb instead:
//!
//! ```ignore
//! acc.seed();
//! acc.mma(&a, &b, Semiring::SUM_PROD);
//! ```
//!
//! An output stating nothing is [`InPlace`](Residence::InPlace) and contracts where it already
//! lies. Both read as those three lines: what differs is the operand's statement, not the kernel.

use cubecl::prelude::*;

use crate::*;

/// An accumulator's scope, opened by [`Tile::accumulate`] and closed by whichever op exhausts it.
/// The variants are the output's [`Residence`] at the level the scope opens, not a choice made
/// here. `EA` is the register accumulate element, distinct from the stored `Out`; an `InPlace`
/// scope has no second element and never reads it.
// The register variant carries a second tile; both are expansion-time handles, so the size gap
// costs nothing the kernel ever sees.
#[allow(clippy::large_enum_variant)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum AccumulatorScope<EA: Numeric, Out: Numeric> {
    /// Contracts in a plane-resident partition, draining into the output on the way out.
    Register {
        tile: Tile<EA>,
        sink: Tile<Out>,
        #[cube(comptime)]
        monoid: Monoid,
    },
    /// Contracts in the output itself, so there is nothing to drain.
    InPlace {
        sink: Tile<Out>,
        #[cube(comptime)]
        monoid: Monoid,
    },
    /// Contracts this instance's share of work the level distributes as one
    /// ([`LevelCuts::distribute`]), which is several of the output's regions whole and part of the
    /// two at either end. The scope is per region of the share rather than per instance, so the
    /// accumulator opens and drains inside the loop and there is none to hold here.
    Distributed {
        sink: Tile<Out>,
        #[cube(comptime)]
        monoid: Monoid,
    },
}

/// A contraction into this accumulation has to add the way the accumulation folds: the scope
/// seeds and drains under its monoid, so a semiring adding under another would merge the plane's
/// partials one way and accumulate the steps another, and the result is neither.
fn adds_the_way_it_folds(monoid: Monoid, semiring: Semiring) {
    assert!(
        semiring.add() == monoid,
        "AccumulatorScope: this accumulation folds under {monoid:?}, so it cannot contract under \
         {semiring:?}"
    );
}

#[cube]
impl<EA: Numeric, Out: Numeric> AccumulatorScope<EA, Out> {
    /// Seed the accumulator with its monoid's identity: `0` under `Sum`, the lowest value under
    /// `Max`, and so on. The identity is minted in whichever element the accumulator actually
    /// holds, which is the element the caller cannot name here.
    pub fn seed(&mut self) {
        match self {
            AccumulatorScope::Register {
                tile,
                sink: _,
                monoid,
            } => tile.init_identity(comptime!(*monoid)),
            AccumulatorScope::InPlace { sink, monoid } => sink.init_identity(comptime!(*monoid)),
            AccumulatorScope::Distributed { sink: _, monoid: _ } => panic!(
                "AccumulatorScope::seed: a distributed scope opens one accumulator per region of \
                 its share and seeds each, so there is nothing here to seed"
            ),
        }
    }

    /// `c = lhs · rhs`: contract into the accumulator and drain, the contraction owning the init.
    /// The scope's whole body at every call site not accumulating onto `c`. `semiring` is the
    /// algebra of the contraction: the product it forms and the monoid those products accumulate
    /// into. The scope holds a fold and nothing more, so a contraction is handed one.
    pub fn mm<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mm(lhs, rhs, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mm(lhs, rhs, semiring)
            }
            AccumulatorScope::Distributed { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                distributed_mm::<EA, Out, Lhs, Rhs>(sink, lhs, rhs, comptime!(*monoid), semiring)
            }
        }
    }

    /// `c += lhs · rhs`: [`mm`](AccumulatorScope::mm) with the accumulate its name carries,
    /// folding onto an accumulator the caller [`seed`](AccumulatorScope::seed)ed. Drains the same
    /// way: the contraction exhausts the scope the accumulator was opened for, so leaving it is
    /// what writes the result back.
    pub fn mma<Lhs: Numeric, Rhs: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mma(lhs, rhs, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mma(lhs, rhs, semiring)
            }
            // The same call as `mm`: a distributed destination is written by instances that
            // cannot see each other, so none of them may claim a cell, and the init every one of
            // them would otherwise do is the launch's ([`Write::Accumulate`]).
            AccumulatorScope::Distributed { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                distributed_mm::<EA, Out, Lhs, Rhs>(sink, lhs, rhs, comptime!(*monoid), semiring)
            }
        }
    }

    /// `c = (lhs ⊗ s) · rhs`, or its rhs twin: [`mm`](AccumulatorScope::mm) with one operand
    /// scaled by a real operand, the side read off the scales' axes
    /// ([`ScaleSide`](crate::ScaleSide)). A register accumulator here is what a decode gemv wants:
    /// the scaled partials never round-trip through the sink between `K` steps.
    pub fn mm_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Sequence<Tile<S>>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mm_scaled(lhs, rhs, scales, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mm_scaled(lhs, rhs, scales, semiring)
            }
            AccumulatorScope::Distributed { sink: _, monoid: _ } => {
                panic!("AccumulatorScope::mm_scaled: a scaled contraction over a distributed share")
            }
        }
    }

    /// `c += (lhs ⊗ s) · rhs`: [`mm_scaled`](AccumulatorScope::mm_scaled) with the accumulate its
    /// name carries, folding onto an accumulator the caller [`seed`](AccumulatorScope::seed)ed.
    pub fn mma_scaled<Lhs: Numeric, Rhs: Numeric, S: Numeric>(
        &mut self,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        scales: &Sequence<Tile<S>>,
        #[comptime] semiring: Semiring,
    ) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                tile.mma_scaled(lhs, rhs, scales, semiring);
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                comptime!(adds_the_way_it_folds(*monoid, semiring));
                sink.mma_scaled(lhs, rhs, scales, semiring)
            }
            AccumulatorScope::Distributed { sink: _, monoid: _ } => {
                panic!(
                    "AccumulatorScope::mma_scaled: a scaled contraction over a distributed share"
                )
            }
        }
    }

    /// `c = fold(input)`: reduce `input` along the axes the accumulator does not span, then
    /// drain, as [`mm`](AccumulatorScope::mm) does. Owns the init the same way.
    pub fn reduce_axis<In: Numeric>(&mut self, input: &Tile<In>) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                tile.reduce_axis(input, comptime!(*monoid));
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                sink.reduce_axis(input, comptime!(*monoid))
            }
            AccumulatorScope::Distributed { sink: _, monoid: _ } => {
                panic!("AccumulatorScope::reduce_axis: a reduction over a distributed share")
            }
        }
    }

    /// `c = fold(c, input)`: [`reduce_axis`](AccumulatorScope::reduce_axis) folding onto an
    /// accumulator the caller [`seed`](AccumulatorScope::seed)ed.
    pub(crate) fn reduce_axis_accumulate<In: Numeric>(&mut self, input: &Tile<In>) {
        match self {
            AccumulatorScope::Register { tile, sink, monoid } => {
                tile.reduce_axis_accumulate(input, comptime!(*monoid));
                tile.drain_cast_into(sink);
            }
            AccumulatorScope::InPlace { sink, monoid } => {
                sink.reduce_axis_accumulate(input, comptime!(*monoid))
            }
            AccumulatorScope::Distributed { sink: _, monoid: _ } => {
                panic!(
                    "AccumulatorScope::reduce_axis_accumulate: a reduction over a distributed share"
                )
            }
        }
    }
}

/// What an accumulator scope opens over: whether the level distributes work as one, read
/// together with the output's [`Residence`] there. Both are needed and neither is a choice made
/// here.
enum Opens {
    /// This instance's own region, in registers.
    Register,
    /// This instance's own region, where it already lies.
    InPlace,
    /// One region at a time of the share this instance holds, in registers. An instance holding a
    /// share of distributed work covers several regions and cannot open one accumulator across
    /// them: the share's length is runtime, and a register block is not.
    PerRegion,
}

/// The scope `residence` opens under this level's distribution, and the refusals that leaves.
fn opens(space: &Space, residence: Residence) -> Opens {
    let distributed = space.partitioner().work().is_some();
    match (residence, distributed) {
        (Residence::Register, false) => Opens::Register,
        (Residence::Register, true) => Opens::PerRegion,
        (Residence::InPlace, false) => Opens::InPlace,
        (Residence::InPlace, true) => panic!(
            "Tile::accumulate: a share of distributed work covers several output regions, and \
             contracting in place would fold into the destination once per step of it rather \
             than once per region; state Residence::Register on the output"
        ),
        (Residence::Smem, _) => panic!(
            "Tile::accumulate: an accumulator has no shared-memory form; state \
             Residence::Register to contract in registers, or nothing to contract in place"
        ),
    }
}

#[cube]
impl<Acc: Numeric> Tile<Acc> {
    /// Open this output's accumulator scope, uninitialized, folding under `monoid`. Stated here
    /// because it is read on drain, when the plane's lanes are combined, and is the accumulation's
    /// one algebra rather than a fact about each call. A fold and nothing more: an op needing more
    /// is handed the [`Semiring`] it runs, whose add this must be.
    ///
    /// The op that closes the scope drains it, and where that op is [`mm`](AccumulatorScope::mm)
    /// or [`reduce_axis`](AccumulatorScope::reduce_axis) it owns the init too; only the
    /// accumulating verbs ask the caller to seed. Where the accumulator lives is the output
    /// operand's own statement ([`Operand::stage`]), and `EA` is read only under
    /// [`Register`](Residence::Register).
    ///
    /// `lhs` states the register form the accumulator takes: a staged cmma or manual-mma operand
    /// meets a matching fragment, and one staging no register form leaves the space's
    /// [`instruction`](Space::instruction) to decide. It also sizes the fragment, since a hardware
    /// fragment is the whole `m × n × k` and an accumulator spans only `m × n`.
    pub fn accumulate<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> AccumulatorScope<EA, Acc> {
        let write = self.write();
        // The stamped value, not a fresh derivation: this space is the operand's own projection
        // and the axis that splits it is exactly the one the projection dropped.
        let split_share = self.split_share();
        comptime!(split_share.validate(write, "Tile::accumulate"));
        let plan = self.stage_plan();
        match comptime!(opens(&self.space, plan.head())) {
            Opens::Register => {
                let tile = self.register_partition::<EA, EL>(lhs, monoid);
                AccumulatorScope::<EA, Acc>::new_Register(tile, self.clone(), monoid)
            }
            Opens::InPlace => AccumulatorScope::<EA, Acc>::new_InPlace(self.clone(), monoid),
            Opens::PerRegion => AccumulatorScope::<EA, Acc>::new_Distributed(self.clone(), monoid),
        }
    }

    /// The plane-resident partition a [`Register`](Residence::Register) scope contracts in,
    /// uninitialized and shaped to meet `lhs` at the instruction.
    pub(crate) fn register_partition<EA: Numeric, EL: Numeric>(
        &self,
        lhs: &Tile<EL>,
        #[comptime] monoid: Monoid,
    ) -> Tile<EA> {
        // The contracted axes are those `lhs` spans and this accumulator does not, and the
        // fragment is sized by their product. Off the leaf space: a caller holds a level above
        // it, while the fragment's own reader (`Tile::fragment_matrix`) is handed the leaf tile
        // directly, and the two edges have to be the same number.
        let k = comptime!(lhs.space.final_space().contracted_extent(&self.space));
        let form = comptime!(self.space.instruction().expect(
            "Tile::accumulate: the register form is the space's statement; add \
             `.instruction(...)` to its tiling"
        ));
        // The block's lines match the memory it drains back into; the hardware
        // encodings ignore it.
        let vector_size = self.vector_size();
        // Not `self.lane_share()`: that is the stamped value, and stamping happens on the way
        // down, after this runs. The space already knows every level, so ask it.
        let lanes = comptime!(self.space.lanes());
        PlanePartition::<EA>::mirror(
            comptime!(self.space.clone()),
            comptime!(MatrixAxes::accumulator(
                &self.space.final_space(),
                &lhs.space.final_space()
            )),
            comptime!(form),
            comptime!(k),
            vector_size,
            lanes,
            monoid,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Buffering, CubeAxis, Tiling, WalkOrder, cubes};
    use cubecl::ir::Scope;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    fn test_scope() -> Scope {
        Scope::root(cubecl::ir::settings::KernelSettings::new(
            cubecl::ir::settings::Dim3::new_single(),
            cubecl::ir::settings::ExecutionMode::Checked,
            cubecl::ir::AddressType::U32,
        ))
    }

    // Opening an accumulator whose cells several instances hold slices of is refused, and the
    // refusal is checked in two halves, each where it can be observed: `space::base` checks the
    // share that is stamped on the tile, and `space::partition::distribution` checks what
    // `validate` does with it. What is left here is one read of the stamped value, which a
    // procedural tile (the only kind that can be built without a launch) does not carry.

    /// A share of distributed work covers several output regions, so an accumulator has to open
    /// per region and a register form is the only one that can. Host-side: a comptime panic raised in a
    /// kernel lands on a worker thread where `#[should_panic]` never sees it.
    #[test]
    #[should_panic = "contracting in place would fold into the destination once per step"]
    fn distributed_work_contracting_in_place_is_refused() {
        let space = Tiling::over(&mut (), &[(M, 8), (N, 8), (K, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.distribute(cubes(CubeAxis::X).instances(3), &[(M, 4), (N, 4), (K, 8)]);
            })
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.walk(&[(M, 4), (N, 4), (K, 4)]);
            })
            .build();
        opens(&space.project(&[M, N]), Residence::InPlace);
    }

    /// The same cut on an axis the output spans is a plain output split: each cube owns its
    /// columns outright, so the accumulator opens as it always has.
    #[test]
    fn opening_an_accumulator_over_a_cube_split_output_is_fine() {
        let scope = test_scope();
        let space = Tiling::over(&mut (), &[(M, 4), (N, 8), (K, 4)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
                l.distribute(cubes(CubeAxis::X), &[(N, 4)])
                    .walk(&[(M, 4), (K, 4)]);
            })
            .build()
            .with_instruction(Instruction::registers(16));
        let out = Tile::<f32>::__expand_zeros(&scope, space.project(&[M, N]));
        let lhs = Tile::<f32>::__expand_zeros(&scope, space.project(&[M, K]));
        out.__expand_accumulate_method::<f32, f32>(&scope, &lhs, Monoid::Sum);
    }
}
